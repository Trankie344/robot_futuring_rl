import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from hil_pico_collection.recording.episode_buffer import SealedEpisode
from hil_pico_collection.web.app import INDEX_PAGE, REPLAY_PAGE, create_app


class Core:
    def __init__(self):
        self._active = None
        self.next_index = 0
        self.drop_count = 0
        self.last_drop_reason = None
        self.last_error = None
        self.current_frame_count = 0
        self.failed_episodes = []

    @property
    def recording(self):
        return self._active is not None

    def start_episode(self, task):
        if self._active is not None:
            raise RuntimeError("an episode is already recording")
        self._active = SimpleNamespace(episode_index=self.next_index, task=task)
        return self._active

    def end_episode(self):
        if self._active is None:
            raise RuntimeError("no episode is recording")
        active = self._active
        self._active = None
        self.next_index += 1
        return SealedEpisode(active.episode_index, active.task, frames=[])


class Writer:
    def __init__(self, root):
        self.root = root
        self.fps = 30
        self.written = []
        self.deleted = []

    def write_episode(self, sealed):
        self.written.append(sealed.episode_index)
        return sealed.episode_index

    def delete_episode(self, index):
        self.deleted.append(index)


def client(tmp_path, reset=None):
    return TestClient(create_app(Core(), Writer(tmp_path), reset_request=reset))


def test_pages_retain_recording_and_video_browsing_without_robot_replay(tmp_path):
    with client(tmp_path, lambda: None) as web:
        assert web.get("/").status_code == 200
        assert web.get("/replay").status_code == 200
    combined = INDEX_PAGE + REPLAY_PAGE
    assert "Dataset video replay" in combined
    assert "Replay Robot" not in combined
    assert "/api/replay/robot" not in combined


def test_start_and_end_enqueue_async_save(tmp_path):
    core = Core()
    writer = Writer(tmp_path)
    with TestClient(create_app(core, writer, reset_request=lambda: None)) as web:
        assert web.post("/api/episodes/start", json={"task": "fold"}).status_code == 200
        assert web.post("/api/episodes/end").status_code == 200
        deadline = time.monotonic() + 1
        while not writer.written and time.monotonic() < deadline:
            time.sleep(0.01)
        assert writer.written == [0]


def test_start_rejects_empty_task(tmp_path):
    with client(tmp_path, lambda: None) as web:
        assert web.post("/api/episodes/start", json={"task": " "}).status_code == 422


def test_end_without_recording_returns_conflict(tmp_path):
    with client(tmp_path, lambda: None) as web:
        assert web.post("/api/episodes/end").status_code == 409


def test_reset_publishes_generic_request_and_returns_accepted(tmp_path):
    calls = []
    with client(tmp_path, lambda: calls.append("reset")) as web:
        response = web.post("/api/robot/reset")
    assert response.json() == {"accepted": True}
    assert calls == ["reset"]


def test_legacy_reset_button_path_uses_same_generic_request(tmp_path):
    calls = []
    with client(tmp_path, lambda: calls.append("reset")) as web:
        response = web.post("/api/robot/reset-arm")
    assert response.json() == {"accepted": True}
    assert calls == ["reset"]


def test_reset_returns_unavailable_without_publisher(tmp_path):
    with client(tmp_path) as web:
        assert web.post("/api/robot/reset").status_code == 503


def test_robot_replay_endpoints_do_not_exist(tmp_path):
    with client(tmp_path, lambda: None) as web:
        assert web.post("/api/replay/episodes/0/robot").status_code == 404
        assert web.post("/api/replay/robot/stop").status_code == 404
