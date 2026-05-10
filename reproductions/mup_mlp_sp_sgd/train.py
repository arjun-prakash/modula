"""SP+SGD-only reproduction of the microsoft/mup CIFAR-10 MLP experiment."""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

try:
    from mup import MuReadout, MuSGD, set_base_shapes
except ImportError as exc:
    raise SystemExit(
        "This reproduction uses the original microsoft/mup layers and optimizer. "
        "Run with `uv run --with mup python reproductions/mup_mlp_sp_sgd/train.py ...` "
        "or install the `mup` package in your environment."
    ) from exc


DEFAULT_WIDTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
LOG_FIELDS = [
    "epoch",
    "train_loss",
    "train_acc",
    "test_loss",
    "test_acc",
    "width",
    "lr",
    "nonlin",
    "criterion",
    "parametrization",
]


class MLP(nn.Module):
    def __init__(
        self,
        width=128,
        num_classes=10,
        nonlin=F.relu,
        output_mult=1.0,
        input_mult=1.0,
        init_std=1.0,
    ):
        super(MLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.init_std = init_std
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = MuReadout(width, num_classes, bias=False, output_mult=output_mult)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode="fan_in")
        self.fc_1.weight.data /= self.input_mult**0.5
        self.fc_1.weight.data *= self.init_std
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode="fan_in")
        self.fc_2.weight.data *= self.init_std
        nn.init.zeros_(self.fc_3.weight)

    def forward(self, x):
        out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        return self.fc_3(out)


def train(args, model, device, train_loader, optimizer, epoch, scheduler=None, criterion=F.cross_entropy):
    model.train()
    train_loss = 0
    correct = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item() * data.shape[0]
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} | ms/batch {:5.2f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    elapsed * 1000 / args.log_interval,
                )
            )
            start_time = time.time()
    if scheduler is not None:
        scheduler.step()
    train_loss /= len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    print(
        "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            train_loss,
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    return train_loss, train_acc


def test(args, model, device, test_loader, evalmode=True, criterion=F.cross_entropy):
    if evalmode:
        model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(data.size(0), -1))
            test_loss += criterion(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss, correct / len(test_loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "SP+SGD-only reproduction of the CIFAR-10 MLP experiment from "
            "microsoft/mup/examples/MLP/main.py."
        )
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--learning_rates", type=float, nargs="*", default=None)
    parser.add_argument("--widths", type=int, nargs="*", default=DEFAULT_WIDTHS)
    parser.add_argument("--output_mult", type=float, default=32.0)
    parser.add_argument("--input_mult", type=float, default=0.00390625)
    parser.add_argument("--init_std", type=float, default=1.0)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--log_interval", type=int, default=300)
    parser.add_argument("--log_dir", type=str, default=".")
    parser.add_argument("--log_file", type=str, default="logs.tsv")
    parser.add_argument("--chart_file", type=str, default="final_train_loss.tsv")
    parser.add_argument("--data_dir", type=str, default="/tmp")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--append_logs", action="store_true")
    return parser.parse_args()


def build_loaders(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    download = not args.no_download
    trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        num_workers=2,
    )
    testset = datasets.CIFAR10(root=args.data_dir, train=False, download=download, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, test_loader


def format_log_value(value):
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6f}"
        return str(value)
    return value


def write_logs(log_path, logs, append=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not append or not log_path.exists()
    mode = "a" if append else "w"
    with log_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, delimiter="\t")
        if should_write_header:
            writer.writeheader()
        for row in logs:
            writer.writerow({field: format_log_value(row[field]) for field in LOG_FIELDS})


def format_log2_lr(lr):
    return f"{math.log2(lr):.6f}".rstrip("0").rstrip(".")


def write_final_train_loss_chart(chart_path, logs):
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    final_by_setting = {}
    for row in logs:
        key = (row["width"], row["lr"])
        if key not in final_by_setting or row["epoch"] >= final_by_setting[key]["epoch"]:
            final_by_setting[key] = row

    widths = sorted({width for width, _ in final_by_setting})
    lrs = sorted({lr for _, lr in final_by_setting})
    lr_columns = [format_log2_lr(lr) for lr in lrs]
    fieldnames = ["width"] + lr_columns

    with chart_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for width in widths:
            chart_row = {"width": width}
            for lr, column in zip(lrs, lr_columns):
                final = final_by_setting.get((width, lr))
                chart_row[column] = "" if final is None else format_log_value(final["train_loss"])
            writer.writerow(chart_row)


def run_setting(args, width, train_loader, test_loader, device):
    logs = []
    mynet = MLP(
        width=width,
        nonlin=torch.relu,
        output_mult=args.output_mult,
        input_mult=args.input_mult,
        init_std=args.init_std,
    ).to(device)
    print("using own shapes")
    set_base_shapes(mynet, None)
    print("done")
    optimizer = MuSGD(mynet.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(args, mynet, device, train_loader, optimizer, epoch, criterion=F.cross_entropy)
        test_loss, test_acc = test(args, mynet, device, test_loader)
        logs.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                width=width,
                lr=args.lr,
                nonlin="relu",
                criterion="xent",
                parametrization="sp",
            )
        )
        if math.isnan(train_loss):
            break
    return logs


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_loader, test_loader = build_loaders(args)
    learning_rates = args.learning_rates if args.learning_rates else [args.lr]

    logs = []
    for lr in learning_rates:
        args.lr = lr
        for width in args.widths:
            print(f"running SP SGD width={width} lr={args.lr}")
            logs.extend(run_setting(args, width, train_loader, test_loader, device))

    log_path = Path(os.path.expanduser(args.log_dir)) / args.log_file
    write_logs(log_path, logs, append=args.append_logs)
    print(log_path)
    chart_path = Path(os.path.expanduser(args.log_dir)) / args.chart_file
    write_final_train_loss_chart(chart_path, logs)
    print(chart_path)


if __name__ == "__main__":
    main()
