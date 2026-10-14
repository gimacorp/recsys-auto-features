"""CLI для загрузки данных и выполнения эксперимента."""

import argparse
import json
from pathlib import Path

from .data import describe, download, load
from .experiment import run


def main():
    parser = argparse.ArgumentParser(description="Автоматический поиск признаков RecSys")
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="Скачать и проверить MovieLens 100K")
    get.add_argument("--data", type=Path, default=Path("data"))
    experiment = commands.add_parser("run", help="Выполнить зафиксированный эксперимент")
    experiment.add_argument("--config", type=Path, default=Path("configs/full.json"))
    experiment.add_argument("--data", type=Path, default=Path("data"))
    experiment.add_argument("--output", type=Path, default=Path("artifacts/full"))
    experiment.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps({"archive": download(args.data), "summary": describe(load(args.data))},
                         ensure_ascii=False, indent=2))
    elif args.command == "run":
        run(args.config, args.data, args.output, args.resume)


if __name__ == "__main__":
    main()
