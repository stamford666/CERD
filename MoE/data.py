import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler
import json
import hashlib
from models import (
    Custom3DCNN,
    IndependentPatchEmbeddings,
    LowRankAdaptivePatchEmbeddings,
    PatchEmbeddings,
)
from torchvision.transforms import Compose, ToTensor, Normalize
import os
import sys
from pathlib import Path
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodal_data import load_abcd_data, resolve_abcd_modalities


ADNI_IMAGE_IMPUTATION_CHOICES = ("legacy_mode", "mean", "median")


def tabular_patch_encoder(args, feature_size):
    """Construct the selected tabular tokenizer without changing defaults."""

    adapter_rank = int(getattr(args, "patch_adapter_rank", 0))
    independent = bool(getattr(args, "independent_patch_embeddings", False))
    if adapter_rank < 0:
        raise ValueError("patch_adapter_rank must be non-negative")
    if adapter_rank > 0 and independent:
        raise ValueError(
            "Low-rank and independent patch encoders are mutually exclusive"
        )
    if adapter_rank > 0:
        return LowRankAdaptivePatchEmbeddings(
            feature_size,
            args.num_patches,
            args.hidden_dim,
            adapter_rank=adapter_rank,
        )
    if independent:
        return IndependentPatchEmbeddings(
            feature_size,
            args.num_patches,
            args.hidden_dim,
        )
    return PatchEmbeddings(feature_size, args.num_patches, args.hidden_dim)


def adni_image_fill_values(
    brain_values,
    train_mask,
    strategy="legacy_mode",
    initial_filling="mean",
):
    """Compute ADNI FreeSurfer fill values from training rows only.

    ``legacy_mode`` preserves the historical ``initial_filling`` behavior
    exactly: the value named ``mean`` used the first pandas mode, while every
    other value used the median.  The explicit ``mean`` and ``median`` modes
    provide unambiguous continuous-feature imputation without consulting any
    validation or test row.
    """

    if strategy not in ADNI_IMAGE_IMPUTATION_CHOICES:
        raise ValueError(
            f"Unsupported ADNI image imputation strategy: {strategy!r}"
        )
    if not isinstance(brain_values, pd.DataFrame) or brain_values.empty:
        raise ValueError("ADNI image imputation requires a non-empty feature frame")
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.ndim != 1 or train_mask.shape[0] != len(brain_values):
        raise ValueError("ADNI image training mask must match the feature rows")
    if not train_mask.any():
        raise ValueError("ADNI image imputation requires non-empty training rows")
    train_brain = brain_values.loc[train_mask]

    if strategy == "legacy_mode" and initial_filling == "mean":
        modes = train_brain.mode(dropna=True)
        fill_values = (
            modes.iloc[0]
            if len(modes)
            else pd.Series(0.0, index=train_brain.columns)
        )
    elif strategy == "mean":
        fill_values = train_brain.mean(axis=0)
    else:
        # Explicit median and the historical non-"mean" path are identical.
        fill_values = train_brain.median(axis=0)

    return fill_values.reindex(train_brain.columns).fillna(0.0)


class MultiModalDataset(Dataset):
    def __init__(self, data_dict, observed_idx, ids, labels, input_dims, transforms, masks, preprocessed=False, use_common_ids=True):
        self.data_dict = data_dict
        self.mc = np.array(data_dict['modality_comb'])
        self.observed = observed_idx
        self.ids = ids
        self.labels = labels
        self.input_dims = input_dims
        self.transforms = transforms
        self.masks = masks
        self.preprocessed = preprocessed
        self.use_common_ids = use_common_ids
        self.data_new = {modality: data[ids] for modality, data in self.data_dict.items() if 'modality' not in modality}
        self.label_new = self.labels[ids]
        self.mc_new = self.mc[ids]
        self.observed_new = self.observed[ids]

        # Sort ids by the number of available modalities
        self.sorted_ids = sorted(np.arange(len(ids)), key=lambda idx: sum([1 for modality in self.data_new if -2 not in self.data_new[modality][idx]]), reverse=True)
        self.data_new = {modality: data[self.sorted_ids] for modality, data in self.data_new.items()}
        self.label_new = self.label_new[self.sorted_ids]
        self.mc_new = self.mc_new[self.sorted_ids]
        self.observed_new = self.observed_new[self.sorted_ids]
        # Optional training-only soft targets are attached by the runner after
        # it has verified an explicit cross-fitted teacher artifact.  Keeping
        # this ``None`` preserves the historical four-item sample exactly for
        # every ordinary training, validation, and test loader.
        self.cross_fitted_teacher_probabilities = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_data = {}
        for modality, data in self.data_new.items():
            sample_data[modality] = data[idx]
            if (modality == 'image') & (not self.preprocessed):
                subj1 = data[idx]
                subj_gm_3d = np.zeros(self.masks.shape, dtype=np.float32)
                subj_gm_3d.ravel()[self.masks] = subj1
                subj_gm_3d = subj_gm_3d.reshape((91, 109, 91))
                if self.transforms:
                    subj_gm_3d = self.transforms(subj_gm_3d)
                sample = subj_gm_3d[None, :, :, :]  # Add channel dimension
                sample_data[modality] = np.array(sample)

        label = self.label_new[idx]
        mc = self.mc_new[idx]
        observed = self.observed_new[idx]

        if self.cross_fitted_teacher_probabilities is None:
            return sample_data, label, mc, observed
        teacher_probabilities = self.cross_fitted_teacher_probabilities[idx]
        return sample_data, label, mc, observed, teacher_probabilities

def convert_ids_to_index(ids, index_map):
    return [index_map[id] if id in index_map else -1 for id in ids]

def load_and_preprocess_image_data(image_path, label_df, id_to_idx):
    # Load and preprocess image data
    image_data = np.load(os.path.join(image_path, 'ADNI_G.npy'), mmap_mode='r')
    mask_path = os.path.join(image_path, 'BLSA_SPGR+MPRAGE_averagetemplate_muse_seg_DS222.nii.gz')

    subject_ids = []
    dates = []
    with open('./data/adni/image/ADNI_subj.txt', 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split('_')
            subject_id = '_'.join(parts[:3])
            date = parts[-1]
            subject_ids.append(subject_id)
            dates.append(date)

    df = pd.DataFrame({
            'PTID': subject_ids,
            'date': dates
        })

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date', ascending=False)
    idx = df.groupby('PTID')['date'].idxmax()

    # Creating the subset DataFrame using the indexes
    subdf = df.loc[idx]
    subdf = subdf.sort_index()
    subdf = subdf.reset_index()

    merged_df = pd.merge(subdf, label_df, on='PTID', how='left')

    image_data = image_data[merged_df['index']]
    final_subject_ids = list(subdf.PTID)

    new_idx = np.array(convert_ids_to_index(final_subject_ids, id_to_idx))
    filtered_idx = [x for x in new_idx if x != -1]
    tmp = np.zeros((len(id_to_idx), image_data.shape[1])) - 2
    tmp[filtered_idx] = image_data[np.array(new_idx) != -1]

    data = nib.load(mask_path).get_fdata()
    mean = image_data.mean()
    std = image_data.std()
    # mean = data.mean()
    # std = data.std()
    mask_gm = (data == 150).ravel()

    return tmp, filtered_idx, mean, std, mask_gm


def load_and_preprocess_adni(args, modality_dict):
    device = getattr(args, 'torch_device', args.device)
    # Paths
    image_path = './data/adni/image'
    preprocessed_image_path = './data/adni/image/UCSFFSX7_06Jan2026.csv'
    genomic_path = './data/adni/genomic/genomic_merged.h5ad'
    clinical_path = './data/adni/clinical/clinical_merged'
    biospecimen_path = './data/adni/biospecimen/biospecimen_merged'
    label_df = pd.read_csv('./data/adni/label.csv', index_col='PTID')
    label_df['DIAGNOSIS'] -= 1
    labels = label_df['DIAGNOSIS'].values.astype(np.int64)
    n_labels = len(set(labels))

    with open('./data/adni/PTID_splits.json') as json_file:
        data_split = json.load(json_file)

    # Preserve the committed split order.  ``set`` made sample order depend on
    # PYTHONHASHSEED and prevented exact replays across independent processes.
    train_ids = list(dict.fromkeys(data_split['training']))
    valid_ids = list(dict.fromkeys(data_split['validation']))
    test_ids = list(dict.fromkeys(data_split['testing']))

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0],4), dtype=bool) # IGCB order

    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if 'I' in args.modality or 'i' in args.modality:
        if args.preprocessed:
            df = pd.read_csv(preprocessed_image_path, low_memory=False)

            # filter the latest record per subject using update_stamp
            df['update_stamp'] = pd.to_datetime(df['update_stamp'], errors='coerce')
            idx = df.groupby('PTID')['update_stamp'].idxmax()
            df = df.loc[idx].reset_index(drop=True)
            df.index = df['PTID']

            # select brain-related features ending with CV, TA, or SV.
            feature_cols = [col for col in df.columns if (
                col.endswith('CV') or col.endswith('TA') or col.endswith('SV')) and col.startswith('ST')
            ]
            df = df[feature_cols]

            # Fit imputation and scaling on training subjects only.  The old
            # path used the complete cohort and leaked validation/test feature
            # distributions into every ADNI experiment.
            brain_values = df.apply(pd.to_numeric, errors='coerce')
            train_mask = brain_values.index.astype(str).isin(set(map(str, train_ids)))
            if not train_mask.any():
                raise ValueError('No ADNI image rows overlap the training split')
            fill_values = adni_image_fill_values(
                brain_values,
                train_mask,
                strategy=getattr(args, 'adni_image_imputation', 'legacy_mode'),
                initial_filling=args.initial_filling,
            )
            brain_filled = brain_values.fillna(fill_values).fillna(0.0)
            scaler = StandardScaler().fit(brain_filled.loc[train_mask].to_numpy(dtype=np.float32))
            arr = scaler.transform(brain_filled.to_numpy(dtype=np.float32))

            new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
            filtered_idx = new_idx[new_idx != -1]
            for idx in filtered_idx:
                update_modality_combinations(idx, 'I')
            tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
            tmp[filtered_idx] = arr[new_idx != -1]
            observed_idx_arr[filtered_idx, modality_dict['image']] = True
            data_dict['image'] = tmp.astype(np.float32)
            common_idx_list.append(set(filtered_idx))
            encoder_dict['image'] = tabular_patch_encoder(args, df.shape[1]).to(device)
            input_dims['image'] = df.shape[1]

        else:
            arr, filtered_idx, mean, std, mask = load_and_preprocess_image_data(image_path, label_df, id_to_idx)
            observed_idx_arr[:, modality_dict['image']] = arr[:, 0] != -2
            for idx in filtered_idx:
                update_modality_combinations(idx, 'I')

            data_dict['image'] = np.array(arr)
            common_idx_list.append(set(filtered_idx))
            encoder_dict['image'] = torch.nn.Sequential(
                Custom3DCNN(hidden_dim=args.hidden_dim).to(device),
                PatchEmbeddings(feature_size=args.hidden_dim, num_patches=args.num_patches, embed_dim=args.hidden_dim).to(device)
                )
            input_dims['image'] = arr.shape[1]
            transforms['image'] = Compose([
                                        ToTensor(),
                                        Normalize(mean=[mean], std=[std]),
                                    ])
            masks['image'] = mask

    if 'G' in args.modality or 'g' in args.modality:
        df = ad.read_h5ad(genomic_path).to_df()
        genomic_index = df.index
        genomic_dim = df.shape[1]
        arr = df.to_numpy(dtype=np.float32, copy=True)
        train_mask = genomic_index.astype(str).isin(set(map(str, train_ids)))
        if not train_mask.any():
            raise ValueError('No ADNI genomic rows overlap the training split')

        # 387k SNPs make a per-column pandas ``apply(mode)`` both slow and
        # memory-heavy.  Genotypes are exactly {0,1,2}; process feature chunks,
        # fill by the training-only mode, and apply a training-only [-1,1]
        # min/max transform in place.
        chunk_size = 16384
        for start in range(0, arr.shape[1], chunk_size):
            stop = min(start + chunk_size, arr.shape[1])
            block = arr[:, start:stop]
            train_block = block[train_mask]
            if args.initial_filling == 'mean':
                counts = np.stack([
                    np.sum(train_block == genotype, axis=0) for genotype in (0.0, 1.0, 2.0)
                ])
                fill_values = counts.argmax(axis=0).astype(np.float32)
            else:
                fill_values = np.nanmedian(train_block, axis=0).astype(np.float32)
                fill_values = np.nan_to_num(fill_values, nan=0.0)
            np.copyto(block, fill_values[None, :], where=np.isnan(block))

            train_filled = block[train_mask]
            train_min = train_filled.min(axis=0)
            train_max = train_filled.max(axis=0)
            span = train_max - train_min
            safe_span = np.where(span > 0, span, 1.0)
            block -= train_min
            block /= safe_span
            block *= 2.0
            block -= 1.0
        new_idx = np.array(convert_ids_to_index(genomic_index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['genomic']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'G')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]

        data_dict['genomic'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['genomic'] = tabular_patch_encoder(args, genomic_dim).to(device)
        input_dims['genomic'] = genomic_dim

    if 'C' in args.modality or 'c' in args.modality:
        path = clinical_path + '.csv'
        df = pd.read_csv(path, index_col=0)
        columns_to_exclude = [col for col in df.columns if col.startswith('PTCOGBEG') or col.startswith('PTADDX') or col.startswith('PTADBEG')]
        if len(columns_to_exclude) > 0:
            df = df.drop(columns_to_exclude, axis=1)
        train_mask = df.index.astype(str).isin(set(map(str, train_ids)))
        if not train_mask.any():
            raise ValueError('No ADNI clinical rows overlap the training split')
        train_frame = df.loc[train_mask]
        fill_values = (
            train_frame.mean(axis=0).fillna(0.0)
            if args.initial_filling == 'mean'
            else train_frame.median(axis=0).fillna(0.0)
        )
        df = df.fillna(fill_values).fillna(0.0)
        arr = df.values.astype(np.float32)
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['clinical']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'C')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]

        data_dict['clinical'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['clinical'] = tabular_patch_encoder(args, df.shape[1]).to(device)
        input_dims['clinical'] = df.shape[1]

    if 'B' in args.modality or 'b' in args.modality:
        path = biospecimen_path + '.csv'
        df = pd.read_csv(path, index_col=0)
        train_mask = df.index.astype(str).isin(set(map(str, train_ids)))
        if not train_mask.any():
            raise ValueError('No ADNI biospecimen rows overlap the training split')
        train_frame = df.loc[train_mask]
        fill_values = (
            train_frame.mean(axis=0).fillna(0.0)
            if args.initial_filling == 'mean'
            else train_frame.median(axis=0).fillna(0.0)
        )
        df = df.fillna(fill_values).fillna(0.0)
        arr = df.values
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['biospecimen']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'B')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]

        data_dict['biospecimen'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['biospecimen'] = tabular_patch_encoder(args, df.shape[1]).to(device)
        input_dims['biospecimen'] = df.shape[1]

    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]
    full_modality_index = min(list(combination_to_index.values()))
    assert (full_modality_index == 0) # max(list(combination_to_index.values()))
    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    if args.use_common_ids:
        common_idxs = set.intersection(*common_idx_list)
        train_idxs = list(common_idxs & set(train_idxs))
        valid_idxs = list(common_idxs & set(valid_idxs))
        test_idxs = list(common_idxs & set(test_idxs))

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return all(data_dict[modality][idx, 0] == -2 for modality in data_dict.keys() if modality != 'modality_comb')

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]

    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index


def resolve_modality_dict(args):
    if args.data.lower() == 'adni':
        return {'image': 0, 'genomic': 1, 'clinical': 2, 'biospecimen': 3}
    if args.data.lower() == 'abcd':
        return resolve_abcd_modalities(args)
    raise ValueError(f"Unsupported dataset: {args.data}")


def load_and_preprocess_data(args, modality_dict):
    if args.data.lower() == 'adni':
        return load_and_preprocess_adni(args, modality_dict)
    if args.data.lower() == 'abcd':
        def encoder_factory(feature_size, _num_patches, _hidden_dim):
            return tabular_patch_encoder(args, feature_size)

        return load_abcd_data(args, modality_dict, encoder_factory)
    raise ValueError(f"Unsupported dataset: {args.data}")

def load_and_preprocess_data_mimic(args, modality_dict):
    device = getattr(args, 'torch_device', args.device)
    # Paths
    lab_path = './data/mimic/lab_x'
    note_path = './data/mimic/note_x'
    code_path = './data/mimic/code_x'
    label_df = pd.read_csv('./data/mimic/labels.csv', index_col='subject_id')
    labels = label_df['one_year_mortality'].values.astype(np.int64)
    n_labels = len(set(labels))

    with open('./data/mimic/PTID_splits_mimic.json') as json_file:
        data_split = json.load(json_file)

    train_ids = list(set(data_split['training']))
    valid_ids = list(set(data_split['validation']))
    test_ids = list(set(data_split['testing']))

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0], args.n_full_modalities), dtype=bool) # IGCB order

    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if 'L' in args.modality or 'l' in args.modality:
        path = lab_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['lab']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'L')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]

        data_dict['lab'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['lab'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(device)
        input_dims['lab'] = arr.shape[1]

    if 'N' in args.modality or 'n' in args.modality:
        path = note_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['note']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'N')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]

        data_dict['note'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['note'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(device)
        input_dims['note'] = arr.shape[1]

    if 'C' in args.modality or 'c' in args.modality:
        path = code_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['code']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'C')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]

        data_dict['code'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['code'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(device)
        input_dims['code'] = arr.shape[1]

    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]
    full_modality_index = min(list(combination_to_index.values()))
    assert (full_modality_index == 0) # max(list(combination_to_index.values()))
    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    if args.use_common_ids:
        common_idxs = set.intersection(*common_idx_list)
        train_idxs = list(common_idxs & set(train_idxs))
        valid_idxs = list(common_idxs & set(valid_idxs))
        test_idxs = list(common_idxs & set(test_idxs))

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return all(data_dict[modality][idx, 0] == -2 for modality in data_dict.keys() if modality != 'modality_comb')

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]

    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index

def collate_fn(batch):
    item_lengths = {len(item) for item in batch}
    if item_lengths == {4}:
        data, labels, mcs, observeds = zip(*batch)
        teachers = None
    elif item_lengths == {5}:
        data, labels, mcs, observeds, teachers = zip(*batch)
    else:
        raise ValueError(
            "A batch must contain uniformly four ordinary fields or five "
            "fields including cross-fitted teacher probabilities"
        )
    modalities = data[0].keys()
    collated_data = {modality: torch.tensor(np.stack([d[modality] for d in data]), dtype=torch.float32) for modality in modalities}
    labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    if teachers is None:
        return collated_data, labels, mcs, observeds
    teacher_probabilities = torch.tensor(
        np.stack(teachers), dtype=torch.float32
    )
    return collated_data, labels, mcs, observeds, teacher_probabilities

def make_data_order_generator(seed):
    """Dedicated generator so sampler order is independent of model RNG."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


class SampleOrderSHARecorder:
    """Training-loop-driven aggregate receipt over dataset positions."""
    algorithm = "sha256-little-u64-dataset-position-v1"
    def __init__(self, seed, train_ids, expected_epochs):
        encoded = b"\0".join(str(item).encode("utf-8") for item in train_ids)
        self.seed, self.expected_epochs = int(seed), int(expected_epochs)
        self.train_inventory_sha256 = hashlib.sha256(encoded).hexdigest()
        self.epoch_receipts, self._active, self._positions = [], None, []

    def begin_epoch(self, epoch, loader_kind):
        if self._active is not None or epoch != len(self.epoch_receipts) + 1:
            raise RuntimeError("sample-order epochs must begin exactly once in sequence")
        self._active, self._positions = (int(epoch), str(loader_kind)), []

    def observe(self, index):
        if self._active is None: raise RuntimeError("sampler iterated outside an active training epoch")
        self._positions.append(int(index))

    def end_epoch(self, epoch):
        if self._active is None or self._active[0] != int(epoch): raise RuntimeError("sample-order epoch end mismatch")
        payload=b"".join(index.to_bytes(8,"little",signed=False) for index in self._positions)
        self.epoch_receipts.append({"epoch":int(epoch),"loader_kind":self._active[1],
          "count":len(self._positions),"dataset_position_order_sha256":hashlib.sha256(payload).hexdigest()})
        self._active, self._positions = None, []

    def receipt(self):
        return {"algorithm":self.algorithm,"seed":self.seed,
          "train_inventory_sha256":self.train_inventory_sha256,
          "expected_epochs":self.expected_epochs,"completed_epochs":len(self.epoch_receipts),
          "epochs":list(self.epoch_receipts)}


class RecordingSampler(Sampler):
    def __init__(self, sampler, recorder):
        self.sampler, self.recorder = sampler, recorder

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for index in self.sampler:
            self.recorder.observe(index)
            yield index


def attach_sample_order_recorder(loader, recorder):
    sampler = loader.batch_sampler.sampler
    if isinstance(sampler, RecordingSampler):
        if sampler.recorder is not recorder: raise RuntimeError("loader has a different order recorder")
    else:
        loader.batch_sampler.sampler = RecordingSampler(sampler, recorder)
    return loader


def create_loaders(data_dict, observed_idx, labels, train_ids, valid_ids, test_ids, batch_size, num_workers, pin_memory, input_dims, transforms, masks, preprocessed, use_common_ids=True, data_order_seed=None, sample_order_recorder=None):
    if ('image' in list(data_dict.keys())) & (not preprocessed):
        train_transfrom = val_transform = test_transform = transforms['image']
        # val_transform = test_transform = False
        mask = masks['image']
    else:
        train_transfrom = val_transform = test_transform = False
        mask = None

    train_dataset = MultiModalDataset(data_dict, observed_idx, train_ids, labels, input_dims, train_transfrom, mask, preprocessed, use_common_ids)
    valid_dataset = MultiModalDataset(data_dict, observed_idx, valid_ids, labels, input_dims, val_transform, mask, preprocessed, use_common_ids)
    test_dataset = MultiModalDataset(data_dict, observed_idx, test_ids, labels, input_dims, test_transform, mask, preprocessed, use_common_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    order_generator = (
        None if data_order_seed is None else make_data_order_generator(data_order_seed)
    )
    if order_generator is None:
        train_loader_shuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    else:
        sampler = RandomSampler(train_dataset, generator=order_generator)
        train_loader_shuffle = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, train_loader_shuffle, val_loader, test_loader

# Updated: full modality index is 0.
def get_modality_combinations(modalities):
    all_combinations = []
    for i in range(len(modalities), 0, -1):
        comb = list(combinations(modalities, i))
        all_combinations.extend(comb)

    # Create a mapping dictionary
    combination_to_index = {''.join(sorted(comb)): idx for idx, comb in enumerate(all_combinations)}
    return combination_to_index
