import hydra
import utils
import wandb
import torch
from tqdm import tqdm
import itertools
from hydra.core.hydra_config import HydraConfig
import os
import gc
import matplotlib.pyplot as plt

@hydra.main(config_path="configs", config_name="config")
def main(cfg): 
    #print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()
    run_name = HydraConfig.get().job.override_dirname
    logger = wandb.init(project="inf473v", name=run_name)
    print (run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    class_names = sorted(os.listdir(cfg.datamodule.train_dataset_path))
    print(class_names)
    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), "checkpoints/" + run_name+"rotnet")
    

    criterion = hydra.utils.instantiate(cfg.criterion)
    
    # rotnet = hydra.utils.instantiate(cfg.rotnet)
    # if cfg.cross_validation == False:
    #     rotnet_train_loader = datamodule.dloader_rotnet()
    # else:
    #     rotnet_train_loader, rotnet_val_loader = datamodule.dloader_rotnet(cfg.cross_validation)


    # optimizer_rotnet = hydra.utils.instantiate(cfg.optimizer, rotnet.parameters())

    # train_rotnet(rotnet, rotnet_train_loader, optimizer_rotnet, criterion, device, cfg.rotnet_epochs, logger, checkpoint_path, val_loader = rotnet_val_loader)
    # rotnet.to_classifier(48)
    # rotnet.to(device)
    # train_loader = datamodule.dloader_labeled()
    # finetune_student(rotnet, cfg.student_epochs, optimizer_rotnet, criterion, device, train_loader, logger, checkpoint_path)



    teacher = hydra.utils.instantiate(cfg.teacher)
    if cfg.self_train == True:
        student = teacher
    else: 
        student = hydra.utils.instantiate(cfg.student)
    optimizer = hydra.utils.instantiate(cfg.optimizer, teacher.parameters())
    optimizer_student = hydra.utils.instantiate(cfg.optimizer, student.parameters())
    train_loader = datamodule.dloader_labeled()

    if cfg.cross_validation == False:
        checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), "checkpoints/" + run_name)
        train_teacher(teacher, train_loader, datamodule, logger, optimizer, criterion, device, cfg.warmup_epochs, cfg.pseudolabeling_epochs, cfg.max_weight, checkpoint_path)
        train_student(teacher, student, train_loader, datamodule, logger, optimizer_student, criterion, device, cfg.student_epochs, checkpoint_path, val_loader)
        student.freeze()
        finetune_student(student, cfg.student_epochs, optimizer_student, criterion, device, train_loader, logger, checkpoint_path)


    else:
        cross_folds = datamodule.generate_folds(5)
        val_acc = [0]*5
        for i, (train_loader, val_loader) in enumerate(cross_folds):
            root = os.path.join(hydra.utils.get_original_cwd(), "checkpoints/")
            checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), "checkpoints/"+run_name+"_"+str(i))
            # get class distribution of train_loader and val_loader
            train_class_distribution = datamodule.get_class_counts(train_loader.dataset)
            val_class_distribution = datamodule.get_class_counts(val_loader.dataset)
            print (train_class_distribution)
            print (val_class_distribution)
            train_teacher(teacher, train_loader, datamodule, logger, optimizer, criterion, device, cfg.warmup_epochs, cfg.pseudolabeling_epochs, cfg.max_weight, checkpoint_path, val_loader)
            print_sample(val_loader, teacher, root, device, class_names)
            print_sample(train_loader, teacher, root, device, class_names)
            train_student(teacher, student, train_loader, datamodule, logger, optimizer_student, criterion, device, cfg.student_epochs, checkpoint_path, val_loader)
            student.freeze()
            finetune_student(student, cfg.student_epochs, optimizer_student, criterion, device, train_loader, logger, checkpoint_path,val_loader= val_loader)
            val_acc[i] = evaluate(student, val_loader, device)
            logger.log({"val_acc": val_acc[i]})
        logger.log({"val_acc_mean": sum(val_acc)/len(val_acc)})
        logger.log({"val_acc_std": sum([(x - sum(val_acc)/len(val_acc))**2 for x in val_acc])/len(val_acc)})


        




def train_teacher(teacher, train_loader, datamodule, logger,  optimizer, criterion, device, warmup_epochs, pseudolabeling_epochs, max_weight, checkpoint_path, val_loader=None):
    teacher.to(device)
    wandb.watch(teacher)
    class_weights = datamodule.get_class_weights(train_loader.dataset).to(device)
    print (class_weights)
    print(class_weights.shape)
    # print (class_weights.dtype)
    for epoch in tqdm(range(warmup_epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for j, batch in tqdm(enumerate(train_loader)):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = teacher(images)
            loss = criterion(preds, labels, weight=class_weights)
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
                "train_acc_epoch": epoch_acc,
            }
        )
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        if val_loader is not None:
            for j, batch in tqdm(enumerate(val_loader)):
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
                    "val_acc_epoch": epoch_acc,
                }
            )


    # save every 10 epochs
        if checkpoint_path is not None:
            if epoch % 10 == 0:
                torch.save(teacher.state_dict(), checkpoint_path+"teacher_epoch_"+str(epoch)+".pth")
    # pseudo-labeling
    # pseudo_loader = None
    # for epoch in tqdm(range(pseudolabeling_epochs)):
    #     if epoch % 10 == 0:
    #         torch.cuda.empty_cache()
    #         pseudo_loader = datamodule.add_labels2(teacher, pseudo_loader, device)
    #         teacher.train()
        
    #     epoch_loss = 0
    #     epoch_num_correct = 0
    #     num_samples = 0
    #     pseudo_loss_weight = max_weight * (epoch / pseudolabeling_epochs)
    #     #print(pseudo_loader.dataset[0])
    #     teacher.to(device)
    #     for j , (batch, pseudo_batch) in tqdm(enumerate(itertools.zip_longest(train_loader, pseudo_loader))):
    #         if pseudo_batch is None:
    #             images, labels = batch
    #             images = images.to(device)
    #             labels = labels.to(device)
    #             preds = teacher(images)
    #             loss = criterion(preds, labels)
    #             epoch_loss += loss.detach().cpu().numpy() * len(images)
    #             epoch_num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
    #             num_samples +=len(images)

    #         elif batch is None:
    #             pseudo_images, pseudo_labels = pseudo_batch
    #             pseudo_images = pseudo_images.to(device)
    #             pseudo_labels = pseudo_labels.type(torch.LongTensor).to(device)
    #             pseudo_preds = teacher(pseudo_images)
    #             loss = pseudo_loss_weight * criterion(pseudo_preds, pseudo_labels)
    #             epoch_loss += loss.detach().cpu().numpy() * len(pseudo_images)
    #             epoch_num_correct += (pseudo_preds.argmax(1) == pseudo_labels).sum().detach().cpu().numpy()
    #             num_samples +=len(pseudo_images)



    #         else: 
    #             images, labels = batch
    #             pseudo_images, pseudo_labels = pseudo_batch
    #             images = images.to(device)
    #             labels = labels.to(device)
    #             pseudo_images = pseudo_images.to(device)
    #             pseudo_labels = pseudo_labels.type(torch.LongTensor).to(device)
    #             #print(type(images))
                
    #             preds = teacher(images)
    #             pseudo_preds = teacher(pseudo_images)
    #             loss = criterion(preds, labels)
    #             # print(f"pseudo_preds type: {pseudo_preds.dtype}")
    #             # print(f"labels type: {labels.dtype}")

    #             pseudo_loss = criterion(pseudo_preds, pseudo_labels)
    #             loss = loss + pseudo_loss_weight * pseudo_loss
    #             epoch_loss += loss.detach().cpu().numpy() * (len (images) + len(pseudo_images))
    #             epoch_num_correct += (
    #                     (preds.argmax(1) == labels).sum().detach().cpu().numpy()
    #                     + (pseudo_preds.argmax(1) == pseudo_labels).sum().detach().cpu().numpy()
    #                 )
    #             num_samples += len(images) + len(pseudo_images)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     epoch_loss /= num_samples
    #     epoch_acc = epoch_num_correct / num_samples
    #     logger.log(
    #         {
    #             "epoch": epoch+ warmup_epochs,
    #             "train_loss_epoch": epoch_loss,
    #             "train_acc": epoch_acc,
    #         }
    #     )



    #     if val_loader is not None:
    #         epoch_loss = 0
    #         epoch_num_correct = 0
    #         num_samples = 0
    #         for j, batch in tqdm(enumerate(val_loader)):
    #             images, labels = batch
    #             images = images.to(device)
    #             labels = labels.to(device)
    #             preds = teacher(images)
    #             loss = criterion(preds, labels)
    #             epoch_loss += loss.detach().cpu().numpy() * len(images)
    #             epoch_num_correct += (
    #                 (preds.argmax(1) == labels).sum().detach().cpu().numpy()
    #             )
    #             num_samples += len(images)
    #         epoch_loss /= num_samples
    #         epoch_acc = epoch_num_correct / num_samples
    #         logger.log(
    #             {
    #                 "epoch": epoch+ warmup_epochs,
    #                 "val_loss_epoch": epoch_loss,
    #                 "val_acc": epoch_acc,
    #             }
    #         )
    #     if epoch % 10 == 0:
    #         torch.save(teacher.state_dict(), checkpoint_path+"teacher_epoch"+str(epoch+warmup_epochs)+".pth")
    # datamodule.reset_labels()

        
def train_student(teacher, student, train_loader, datamodule, logger, optimizer, criterion, device, epochs, checkpoint_path, val_loader=None):
    teacher.to(device)
    student.to(device)
    wandb.watch(student)
    # get the teacher to label all the unlabeled data
    
    #print(train_loader.dataset[0][1])

    train_loader = datamodule.label_all(teacher, train_loader,device)
    
    #print(train_loader.dataset[1000][1])
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for j, batch in tqdm(enumerate(train_loader)):
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
                "epoch_student": epoch,
                "train_loss_epoch_student": epoch_loss,
                "train_acc_student": epoch_acc,
            }
        )
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        if val_loader is not None:
            for j, batch in tqdm(enumerate(val_loader)):
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

    for epoch in tqdm(range(finetune_epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for j, batch in tqdm(enumerate(train_loader)):
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
            for j, batch in tqdm(enumerate(val_loader)):
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
    for j, batch in tqdm(enumerate(val_loader)):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
        num_samples += len(images)
    return num_correct / num_samples

def train_rotnet(model, train_loader, optimizer, criterion, device, epochs, logger, checkpoint_path, val_loader=None):
    # print memory usage before loading the model
    #print(torch.cuda.memory_summary(device=device))
    model.to(device)
    #print(torch.cuda.memory_summary(device=device))
    model.train()
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for j, batch in tqdm(enumerate(train_loader)):

            batch = batch.to(device)
            batch = utils.generate_rotations(batch, device)
            images, labels = batch
            #print(torch.cuda.memory_summary(device=device))
            # images = images.to(device)
            # labels = labels.to(device)
            #print(torch.cuda.memory_summary(device=device))

            preds = model(images)
            # print (preds[0])
            #expect a 1x4 tensor of probabilities
            # print (labels[0])
            # expect an int between 0 and 3
            # print(labels[0:4])
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * len(images.cpu())

            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images.cpu())
            if j % 100 == 0:
                # clear memory every 100 batches
                #os.system("nvidia-smi")
                gc.collect()
                torch.cuda.empty_cache()
                #os.system("nvidia-smi")
                logger.log({"rot_train_loss": loss.detach().cpu().numpy()})

        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "rot_epoch": epoch,
                "rot_train_loss_epoch": epoch_loss,
                "rot_train_acc": epoch_acc,
            }
        )
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        if val_loader is not None:
            for j, batch in tqdm(enumerate(val_loader)):
                batch = utils.generate_rotations(batch,device)
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
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
                    "rot_epoch": epoch,
                    "rot_val_loss_epoch": epoch_loss,
                    "rot_val_acc": epoch_acc,
                }
            )
    torch.save(model.state_dict(), checkpoint_path+"rot.pth"+str(epochs))
    # save the model


def print_sample(dataloader, model, path, device, class_names):
    # print a sample of the images and their predictions as title of the image, to check if the model is working
    
    
    model.eval()
    model.to(device)
    image, label = next(iter(dataloader))
    image = image.to(device)
    label = label.to(device)
    pred = model(image)
    pred = pred.argmax(1)
    # unnormalize the image
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)

    image = image.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()


    filename = path + class_names[label[0]] + "_" + class_names[pred[0]] + ".png"
    # save image to file
    # make sure the image is in the right format
    # RGB values between 0 and 1
    #
    plt.imsave(filename, image[0].transpose(1, 2, 0))

    return

if __name__ == "__main__":
    main()
