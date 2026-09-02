import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_runner_does_not_write_individual_level_numpy_artifacts():
    tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
    numpy_writes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (
            isinstance(owner, ast.Name)
            and owner.id == "np"
            and node.func.attr in {"save", "savez", "savez_compressed"}
        ):
            numpy_writes.append(node.func.attr)
    assert numpy_writes == []


def test_public_run_record_omits_history_and_raw_config():
    source = (ROOT / "train.py").read_text(encoding="utf-8")
    assert '"history":' not in source
    assert '"config":' not in source
    assert '"validation_metrics":' not in source


def test_checkpoint_metadata_uses_sanitized_protocol_allowlist():
    source = (ROOT / "train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    checkpoint_saves = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "save"
    ]
    assert len(checkpoint_saves) == 1
    payload = checkpoint_saves[0].args[0]
    assert isinstance(payload, ast.Dict)
    keys = [key.value for key in payload.keys if isinstance(key, ast.Constant)]
    assert "args" in keys
    protocol_value = payload.values[keys.index("args")]
    assert isinstance(protocol_value, ast.Call)
    assert isinstance(protocol_value.func, ast.Name)
    assert protocol_value.func.id == "public_protocol_parameters"


def test_protocol_metadata_and_console_result_omit_local_paths():
    source = (ROOT / "train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "PUBLIC_PROTOCOL_PARAMETER_NAMES"
            for target in node.targets
        )
    )
    allowed = set(ast.literal_eval(assignment.value))
    assert allowed.isdisjoint(
        {"dataset_manifest", "adni_data_root", "output_dir", "device"}
    )
    assert '"result": str(result_path)' not in source
    assert '"result": result_path.name' in source
