from utils import (
    DEVICE,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    DataLoader,
    torch,
    optim,
    nn,
    tqdm,
    PeakSignalNoiseRatio,
)
from dataset import GoProDataset
from models import DeblurringSimple
from visualize import plot_one_image_comparison, plot_kernel_comparison, compare_models


def train_epoch(loader, model, optimizer, loss_fn, loop):
    epoch_loss = 0.0
    for data_blur, data_sharp in loop:
        data_blur = data_blur.to(DEVICE)
        data_sharp = data_sharp.to(DEVICE)

        outputs = model(data_blur)
        loss = loss_fn(outputs, data_sharp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(train_loss=loss.item())

    return epoch_loss / len(loader)


def validate(val_loader, model, loss_fn, eval_metric):
    model.eval()
    loop = tqdm(val_loader, desc=f"Validation", leave=True)
    epoch_loss = 0.0
    psnr_sum = 0.0
    with torch.no_grad():
        for data_blur, data_sharp in loop:
            data_blur = data_blur.to(DEVICE)
            data_sharp = data_sharp.to(DEVICE)

            outputs = model(data_blur)
            loss = loss_fn(outputs, data_sharp)

            epoch_loss += loss.item()
            psnr_sum += eval_metric(outputs, data_sharp).item()
            loop.set_postfix(val_loss=loss.item())

    avg_val_loss = epoch_loss / len(val_loader)
    avg_psnr = psnr_sum / len(val_loader)
    return avg_val_loss, avg_psnr


def test(test_loader, model, loss_fn, eval_metric):
    print("\nRunning Final Test Evaluation...")
    model.eval()
    loop = tqdm(test_loader, desc=f"Testing", leave=True)
    test_loss = 0.0
    psnr_sum = 0.0

    with torch.no_grad():
        for data_blur, data_sharp in loop:
            data_blur = data_blur.to(DEVICE)
            data_sharp = data_sharp.to(DEVICE)

            outputs = model(data_blur)
            loss = loss_fn(outputs, data_sharp)

            test_loss += loss.item()
            psnr_sum += eval_metric(outputs, data_sharp).item()

    avg_test_loss = test_loss / len(test_loader)
    avg_psnr = psnr_sum / len(test_loader)

    print(f"TEST RESULTS:")
    print(f"  Average Test Loss: {avg_test_loss:.4f}")
    print(f"  Average Test PSNR: {avg_psnr:.2f} dB")
    return avg_psnr


def full_training_loop(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    best_val_metric = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        print()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]", leave=True)

        avg_train_loss = train_epoch(train_loader, model, optimizer, loss_fn, loop)
        val_loss, val_metric = validate(val_loader, model, loss_fn, psnr_metric)

        print(f"  Average Train Loss: {avg_train_loss:.4f}")
        print(f"  Average Val Loss: {val_loss:.4f}")
        print(f"  Val PSNR: {val_metric:.2f} dB")

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), f"best_model_{model.__class__.__name__}.pth")
            print("    New best PSNR")

    print("Training completed.")


### experiments
def experiment_1(model: nn.Module, train_loader, val_loader, test_loader):
    print("\nExperiment 1")

    full_training_loop(model, train_loader, val_loader)

    # load best model for testing
    model.load_state_dict(torch.load(f"best_model_{model.__class__.__name__}.pth"))
    print("Loaded best model for testing.")

    # run Test
    loss_fn = nn.MSELoss()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    test(test_loader, model, loss_fn, psnr_metric)

    # visual Analysis
    plot_one_image_comparison(model, test_loader, psnr_metric, "Test Set Result")
    plot_kernel_comparison(model)


# TODO add more experiments
def experiment_2(model: nn.Module, train_loader, val_loader, test_loader):
    print("\nExperiment 2")
    # TODO implement experiment 2


def experiment_3(model: nn.Module, train_loader, val_loader, test_loader):
    print("\nExperiment 3")
    # TODO implement experiment 3


def show_last_best(model: nn.Module, test_loader):
    print("Loaded last best model")

    model.load_state_dict(torch.load(f"best_model_{model.__class__.__name__}.pth"))
    eval_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    # visual Analysis
    plot_one_image_comparison(model, test_loader, eval_metric, "Best model image comparison")
    plot_kernel_comparison(model)


if __name__ == "__main__":
    ### model and dataset selection
    model = DeblurringSimple().to(DEVICE)
    dataset = GoProDataset

    ### dataset and loaders
    train_dataset = dataset(split="train", image_size=IMAGE_SIZE, augment=True)
    val_dataset = dataset(split="val", image_size=IMAGE_SIZE, augment=False)
    test_dataset = dataset(split="test", image_size=IMAGE_SIZE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ### experiments
    # experiment_1(model, train_loader, val_loader, test_loader)
    # TODO add more experiments

    ### show the best result
    # show_last_best(model, test_loader)
    
    ### compare all models
    compare_models(test_loader)
