#!/usr/bin/env python3
"""Box-local LocalShred entry-service -> precursor-gate fire loop (Nuremberg measurement).

This is the ~3 lines of box-local gRPC wiring that the repo test suite can't exercise (it needs
the live proxy + the grpc_tools-generated stubs). It is deliberately NOT imported by anything in
the package: grpcio, the generated `*_pb2` stubs and solders are box-only, so every box-only
import lives inside main() and this file never affects `pytest` collection.

Flat box layout (one directory, scp'd from the repo + generated on the box):
    localshred_consumer.py   localshred_tap.py   frontrun_live.py   precursor_gate.py   (repo)
    shared_pb2.py   shredstream_pb2.py   shredstream_pb2_grpc.py                         (protoc)
    gate.json                                                                            (exported)
    run_localshred.py                                                                    (this file)
The 4 repo modules import flat here (their `try: from . import … except ImportError: import …`
guard), so no package install is needed -- just run this file from that directory.

Prereqs (see SHRED_DECODER.md "Nuremberg measurement runbook"):
  1. the LocalShred proxy running with --grpc-service-port 9898 (the Docker container)
  2. ~/sv/bin/pip install grpcio grpcio-tools solders   (solders already in the live venv)
  3. the three *_pb2*.py stubs generated next to this file:
       python -m grpc_tools.protoc -I jito_protos/protos \
         --python_out=. --grpc_python_out=. shared.proto shredstream.proto

Run:
    cd <flat dir>
    ~/sv/bin/python -u run_localshred.py --gate gate.json --out ls-fires.jsonl
Then ship ls-fires.jsonl to Fly and `frontrun-verify` it against the box-exact 8/30 baseline.
"""

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="LocalShred entry service -> precursor-gate fires / feature capture")
    ap.add_argument("--gate", required=True, help="precursor gate JSON (the direct-view fit)")
    ap.add_argument("--out", default="ls-fires.jsonl", help="append each fire here, box shape")
    ap.add_argument("--addr", default="127.0.0.1:9898", help="proxy --grpc-service-port endpoint")
    ap.add_argument("--tier", default="1", help="gate tier: '1' top one percent, '2' top two percent")
    ap.add_argument("--fresh-window-slots", type=int, default=450,
                    help="freshness gate: fire only on pools first seen after a warm-up of this many "
                         "slots and within this many slots of that (~450 = ~180s); 0 disables it")
    ap.add_argument("--capture", metavar="PATH", default=None,
                    help="also capture our decode's per-slot features of every FRESH pool to PATH (JSONL), to fit "
                         "the study on our own feed: run a day, then localshred-merge + flurry-study --frontrun --flows")
    ap.add_argument("--span-s", type=float, default=180.0, help="--capture: seconds of each pool's life to record")
    ap.add_argument("--warmup-s", type=float, default=180.0,
                    help="--capture: skip pools first seen in this warm-up (they may be mid-life)")
    args = ap.parse_args()

    # Box-only deps: imported here so the file stays safe to have in the repo / import-scan.
    import grpc  # noqa: WPS433

    # The generated stubs sit flat next to this file; make sure that dir is importable.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import shredstream_pb2 as pb  # noqa: WPS433
    import shredstream_pb2_grpc as pbg  # noqa: WPS433

    import localshred_consumer as lc  # noqa: WPS433
    import localshred_tap as tap  # noqa: WPS433

    gate = lc.load_gate(args.gate)
    writer = lc.jsonl_writer(args.out)
    n = {"fires": 0}

    def on_fire(sig):
        writer(sig)
        n["fires"] += 1
        line = lc.fire_line(sig)
        print(f"[fire #{n['fires']}] {line}", flush=True)

    cap = None
    capf = None
    if args.capture:
        capf = open(args.capture, "a")
        cw = {"n": 0}

        def _cap_write(rec):
            capf.write(json.dumps(rec) + "\n")
            capf.flush()
            cw["n"] += 1
            if cw["n"] % 50 == 0:
                print(f"[capture] {cw['n']} pools written -> {args.capture}", flush=True)

        cap = lc.CaptureSink(_cap_write, span_s=args.span_s, warm_s=args.warmup_s)

    consumer = lc.Consumer(gate, tier=args.tier, on_fire=on_fire,
                           fresh_window_slots=args.fresh_window_slots,
                           on_capture=cap.on_capture if cap else None)

    channel = grpc.insecure_channel(args.addr)
    stub = pbg.ShredstreamProxyStub(channel)
    warm = "off" if args.fresh_window_slots <= 0 else f"{args.fresh_window_slots} slots (~{args.fresh_window_slots*0.4:.0f}s warm-up, no fires until then)"
    capnote = f" capture={args.capture} (span {args.span_s:.0f}s, warm {args.warmup_s:.0f}s)" if args.capture else ""
    print(
        f"[localshred] subscribing addr={args.addr} gate={args.gate} tier={args.tier} "
        f"fresh_window={warm} out={args.out}{capnote}",
        flush=True,
    )
    try:
        stream = stub.SubscribeEntries(pb.SubscribeEntriesRequest())
        tap.consume_stream(stream, consumer)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.flush_all()
        if cap:
            cap.close()
            print(f"[capture] {cap.written} pools total -> {args.capture}", flush=True)
            if capf:
                capf.close()
        print(f"[localshred] stopped after {n['fires']} fires", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
