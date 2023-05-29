import hydra
import utils
import wandb
import torch
from tqdm import tqdm
import itertools
from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path="configs", config_name="config")
def main(cfg): 
    run_name = HydraConfig.get().job.override_dirname
    logger = wandb.init(project="inf473v", name=run_name)
    print (run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    return
    teacher = hydra.utils.instantiate(cfg.teacher)
    optimizer = hydra.utils.instantiate(cfg.optimizer, teacher.parameters())
    criterion = hydra.utils.instantiate(cfg.criterion)
    if cfg.self_train == True:
        student = teacher
    else: 
        student = hydra.utils.instantiate(cfg.student)

    train_loader = datamodule.dloader_labeled()

    if cfg.cross_validation == False:
        checkpoint_path = "checkpoints/" + run_name
        train_teacher(teacher, train_loader, datamodule, logger, optimizer, criterion, device, cfg.warmup_epochs, cfg.pseudolabeling_epochs, cfg.max_weight, checkpoint_path)
        train_student(student, train_loader, datamodule, logger, optimizer, criterion, device, cfg.warmup_epochs, cfg.pseudolabeling_epochs, cfg.max_weight, checkpoint_path)
        finetune_student(student, train_loader, datamodule, logger, optimizer, criterion, device, cfg.warmup_epochs, cfg.pseudolabeling_epochs, cfg.max_weight, checkpoint_path)
    else:
        cross_folds = datamodule.generate_folds(5)
        val_acc = [0]*5
        for i, (train_loader, val_loader) in enumerate(cross_folds):
            checkpoint_path = "checkpoints/" + run_name + "_" + str(i)
            train_teacher(teacher, train_loader, datamodule, logger, optimizer, criterion, device, cfg.warmup_epochs, cfg.pseudolabeling_epochs, cfg.max_weight, checkpoint_path, val_loader)
            train_student(student, train_loader, datamodule, logger, optimizer, criterion, device, cfg.warmup_epochs, cfg.pseudolabeling_epochs, cfg.max_weight, checkpoint_path, val_loader)
            finetune_student(student, train_loader, datamodule, logger, optimizer, criterion, device, cfg.warmup_epochs, cfg.pseudolabeling_epochs, cfg.max_weight, checkpoint_path, val_loader)
            val_acc[i] = evaluate(student, val_loader, device)
            logger.log({"val_acc": val_acc[i]})
        logger.log({"val_acc_mean": sum(val_acc)/len(val_acc)})
        logger.log({"val_acc_std": sum([(x - sum(val_acc)/len(val_acc))**2 for x in val_acc])/len(val_acc)})


        




def train_teacher(teacher, train_loader, datamodule, logger,  optimizer, criterion, device, warmup_epochs, pseudolabeling_epochs, max_weight, checkpoint_path, val_loader=None):
    teacher.to(device)
    wandb.watch(teacher)
    for epoch in tqdm(range(warmup_epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for j, batch in enumerate(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = teacher(images)
            loss = criterion(preds, labels)
            logger.log({"loss": loss.detach().cpu().numpy()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        if val_loader is not None:
            for j, batch in enumerate(val_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = teacher(images)
                loss = criterion(preds, labels)
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (
                    (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                )
                num_samples += len(images)
            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            logger.log(
                {
                    "epoch": epoch,
                    "val_loss_epoch": epoch_loss,
                    "val_acc": epoch_acc,
                }
            )
    # save every 10 epochs
        if checkpoint_path is not None:
            if epoch % 10 == 0:
                torch.save(teacher.state_dict(), checkpoint_path+"teacher_epoch_"+str(epoch)+".pth")
    # pseudo-labeling
    pseudo_loader = None
    for epoch in tqdm(range(pseudolabeling_epochs)):
        pseudo_loader = datamodule.add_labels(teacher, pseudo_loader, device)
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        pseudo_loss_weight = max_weight * (epoch / pseudolabeling_epochs)
        loader = itertools.zip_longest(train_loader, pseudo_loader)
        for j, (batch, pseudo_batch) in loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            pseudo_images, pseudo_labels = pseudo_batch
            pseudo_images = pseudo_images.to(device)
            pseudo_labels = pseudo_labels.to(device)
            preds = teacher(images)
            pseudo_preds = teacher(pseudo_images)
            loss = criterion(preds, labels) + pseudo_loss_weight * criterion(pseudo_preds, pseudo_labels)
            logger.log({"loss": loss.detach().cpu().numpy()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * (len(images) + len(pseudo_images))
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                + (pseudo_preds.argmax(1) == pseudo_labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images) + len(pseudo_images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch+ warmup_epochs,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )


        if val_loader is not None:
            epoch_loss = 0
            epoch_num_correct = 0
            num_samples = 0
            for j, batch in enumerate(val_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = teacher(images)
                loss = criterion(preds, labels)
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (
                    (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                )
                num_samples += len(images)
            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            logger.log(
                {
                    "epoch": epoch+ warmup_epochs,
                    "val_loss_epoch": epoch_loss,
                    "val_acc": epoch_acc,
                }
            )
        if epoch % 10 == 0:
            torch.save(teacher.state_dict(), checkpoint_path+"teacher_epoch"+str(epoch+warmup_epochs)+".pth")
    datamodule.reset_labels()

        
def train_student(teacher, student, train_loader, datamodule, logger, optimizer, criterion, device, epochs, checkpoint_path, val_loader=None):
    teacher.to(device)
    student.to(device)
    wandb.watch(student)
    # get the teacher to label all the unlabeled data
    train_loader = datamodule.add_labels(teacher, train_loader, device)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for j, batch in enumerate(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = student(images)
            loss = criterion(preds, labels)
            logger.log({"loss": loss.detach().cpu().numpy()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        if val_loader is not None:
            for j, batch in enumerate(val_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = student(images)
                loss = criterion(preds, labels)
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (
                    (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                )
                num_samples += len(images)
            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            logger.log(
                {
                    "epoch": epoch,
                    "val_loss_epoch": epoch_loss,
                    "val_acc": epoch_acc,
                }
            )
        if checkpoint_path is not None:
            if epoch % 10 == 0:
                torch.save(student.state_dict(), checkpoint_path+"student_epoch_"+str(epoch)+".pth")
    
def finetune_student(student, finetune_epochs, optimizer, criterion, device, train_loader, logger, checkpoint_path, val_loader=None):
    student.train()
    # freeze all but the last layers
    for param in student.parameters():
        param.requires_grad = False
    for param in student.classifier.parameters():
        param.requires_grad = True
    
    for epoch in tqdm(range(finetune_epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for j, batch in enumerate(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = student(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        if val_loader is not None:
            for j, batch in enumerate(val_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = student(images)
                loss = criterion(preds, labels)
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (
                    (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                )
                num_samples += len(images)
            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            logger.log(
                {
                    "epoch": epoch,
                    "val_loss_epoch": epoch_loss,
                    "val_acc": epoch_acc,
                }
            )
    # save the model
    torch.save(student.state_dict(), checkpoint_path+"student_finetuned.pth")

def evaluate(model, val_loader, device):
    # return the accuracy of the model on the validation set
    model.eval()
    num_correct = 0
    num_samples = 0
    for j, batch in enumerate(val_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
        num_samples += len(images)
    return num_correct / num_samples





if __name__ == "__main__":
    main()
