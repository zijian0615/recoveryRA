"""Serve the ACT checkpoint for the Isaac eval. Does not move the real robot."""
import argparse
import pickle
import socket

import packaging.version  # LeRobot's checkpoint loader accesses packaging.version directly.
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.to(args.device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(1)
    print(f"[act] listening {args.port}", flush=True)
    while True:
        conn, _ = server.accept()
        with conn:
            while True:
                header = conn.recv(8)
                if not header:
                    break
                if len(header) != 8:
                    break
                n = int.from_bytes(header, "big")
                buf = b""
                while len(buf) < n:
                    chunk = conn.recv(n - len(buf))
                    if not chunk:
                        break
                    buf += chunk
                if len(buf) != n:
                    break
                msg = pickle.loads(buf)
                if msg["op"] == "reset":
                    policy.reset()
                    reply = {"ok": True}
                else:
                    batch = {}
                    for key, image in msg["images"].items():
                        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        batch[f"observation.images.{key}"] = tensor
                    state = torch.from_numpy(msg["state"]).float().unsqueeze(0)
                    batch["observation.state"] = state
                    batch["task"] = msg["task"]
                    batch = pre(batch)
                    with torch.inference_mode():
                        action = policy.select_action(batch)
                    action = post(action)
                    if isinstance(action, dict):
                        action = action[ACTION]
                    reply = {"action": action.detach().float().cpu().numpy().reshape(-1)}
                payload = pickle.dumps(reply)
                conn.sendall(len(payload).to_bytes(8, "big") + payload)


if __name__ == "__main__":
    main()
