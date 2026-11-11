"""Identify the OBS device by its identity even when its display name has changed."""

import sys
import uuid

from .windows_camera_names import OBS_GUID

_known_ids = set()


def obs_id(identifier):
    raw = bytes(identifier)
    guid = uuid.UUID(OBS_GUID)
    if guid.bytes_le in raw or guid.bytes in raw:
        return True
    for encoding in ("utf-8", "utf-16-le"):
        text = raw.decode(encoding, errors="ignore").lower()
        if str(guid) in text or "obs-studio.virtualcam" in text:
            return True
    return False


def registered_names():
    if sys.platform != "win32":
        return ()
    from .windows_camera_names import CameraNameError, WindowsRegistry

    try:
        return WindowsRegistry().snapshot()
    except (CameraNameError, OSError):
        return ()


def is_obs_camera(device, entries=()):
    if bytes(device.id()) in _known_ids or obs_id(device.id()):
        return True
    name = device.description().strip().casefold()
    if name == "obs virtual camera" or name == "obs camera":
        return True
    # Some capture backends provide a name rather than a GUID in their device identifier.
    return any(name == r.friendly.casefold() for r in entries)


def remember_obs_devices(devices, entries=()):
    devices = list(devices)
    for device in devices:
        raw = bytes(device.id())
        name = device.description().strip().casefold()
        unique_name = sum(d.description().strip().casefold() == name for d in devices) == 1
        if raw and (obs_id(raw) or (unique_name and is_obs_camera(device, entries))):
            _known_ids.add(raw)


def camera_label(device, entries=()):
    if entries and is_obs_camera(device, entries):
        preferred = next((r for r in entries if r.view == 64), entries[0])
        return preferred.friendly
    return device.description()
