"""CLI wiring smoke tests — no training, just argparse + entry point."""


def test_help_lists_train(cli):
    result = cli("--help")
    assert result.returncode == 0
    assert "train" in result.stdout


def test_version(cli):
    result = cli("--version")
    assert result.returncode == 0
    assert "stemlm 0.1.0" in result.stdout


def test_train_help(cli):
    result = cli("train", "--help")
    assert result.returncode == 0
    assert "--output_dir" in result.stdout


def test_no_subcommand_errors(cli):
    result = cli()
    assert result.returncode != 0


def test_unknown_subcommand_errors(cli):
    result = cli("bogus")
    assert result.returncode != 0
