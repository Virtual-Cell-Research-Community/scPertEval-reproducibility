from workflows.protocol_behavior_env import PROTOCOL_PROJECT, build_command, protocol_environment


def test_protocol_behavior_command_uses_the_isolated_project(monkeypatch):
    monkeypatch.setattr("workflows.protocol_behavior_env.shutil.which", lambda _name: "/usr/bin/uv")

    command = build_command(["--list-stages"])

    assert command[:4] == ["/usr/bin/uv", "run", "--project", str(PROTOCOL_PROJECT)]
    assert command[-3:] == ["-m", "analyses.protocol_behavior.cli", "--list-stages"]


def test_protocol_environment_does_not_reuse_the_root_uv_environment():
    environment = protocol_environment(
        {
            "UV_PROJECT_ENVIRONMENT": "/tmp/root",
            "VIRTUAL_ENV": "/tmp/root",
            "SCPERTEVAL_PROTOCOL_ENVIRONMENT": "/tmp/protocol",
            "KEEP": "yes",
        }
    )

    assert environment["UV_PROJECT_ENVIRONMENT"] == "/tmp/protocol"
    assert environment["KEEP"] == "yes"
    assert "SCPERTEVAL_PROTOCOL_ENVIRONMENT" not in environment
    assert "VIRTUAL_ENV" not in environment
