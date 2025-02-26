import torch
from matplotlib import pyplot as plt
import numpy as np
import os
from utils import loss_func_from_target_sigma

def train(model, train_loader=None, test_loader=None, optimizer=None, scheduler=None, epochs=1, 
          prior_sigma=None, prior_logprob=None,
          target_sigma=None, loss_fn=None,
          device="cpu", logger_info=None,
          plot=False, plotpath=None, verbose = False, print_every_epoch=1):
    if logger_info == None: logger_info=print
    epoch_test_losses = []
    epoch_train_losses = []
    all_train_losses = []
    epoch_test_accuracy = []
    epoch_train_accuracy =[]
    epoch_pred_losses = []
    epoch_train_grad = []
    step_train_grad = []
    lrs = []
    model.to(device)
    batch_number = 0
    train_loss_batch_number =[]
    test_loss_batch_number = []
    
    num_batches_train = len(train_loader)
    total_obs_train = len(train_loader.dataset)
    epoch=0
    sgd_tracing_done = False
    while (epoch < epochs):
        epoch += 1
        epoch_train_loss = 0
        epoch_pred_loss = 0
        #current_correct_num = 0
        epoch_sumsqr_gradient = 0
        for x, y in train_loader:
            batch_number += 1
            optimizer.zero_grad()
            model.train()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            if prior_sigma != None:
                if prior_sigma == 0:
                    regularization_loss = 0
                else:
                    regularization_loss = torch.nn.functional.mse_loss(torch.nn.utils.parameters_to_vector(model.parameters()), torch.zeros_like(torch.nn.utils.parameters_to_vector(model.parameters())), reduction="sum")/(2*prior_sigma**2)
            else:
                regularization_loss = -prior_logprob(torch.nn.utils.parameters_to_vector(model.parameters()))
            
            loss_fn = loss_func_from_target_sigma(loss_fn, target_sigma)
            
            pred_loss = loss_fn(pred, y)/x.shape[0]  # we use sum-loss, and divide by the number of observations in batch
            loss = pred_loss + regularization_loss/total_obs_train  # loss is now a per-observation loss 
            loss.backward()
            epoch_train_loss += loss.item() / num_batches_train
            epoch_pred_loss += pred_loss.item() / num_batches_train
            # pred_class = torch.argmax(pred, dim=-1)
            # current_correct_num += (pred_class == y).sum()
            all_train_losses.append(loss.item())
            train_loss_batch_number += [batch_number]

            sumsqr_gradient = sum([(p.grad**2).sum() for p in model.parameters()]).item()
            step_train_grad.append(sumsqr_gradient)
            epoch_sumsqr_gradient += sumsqr_gradient

            lrs.append(optimizer.param_groups[0]['lr'].item())
            optimizer.step()
        if scheduler:
            scheduler.step()
        #train_accuracy = current_correct_num.item() / total_obs_train
        #epoch_train_accuracy.append(train_accuracy)

        epoch_train_losses.append(epoch_train_loss)
        epoch_pred_losses.append(epoch_pred_loss)

        epoch_train_grad.append(epoch_sumsqr_gradient/num_batches_train)

        epoch_test_loss = 0
        current_correct_num = 0
        total_obs_test = len(test_loader.dataset)
        num_batches_test = len(test_loader)
        if prior_sigma != None:
            if prior_sigma == 0:
                regularization_loss = 0
            else:
                regularization_loss = torch.nn.functional.mse_loss(torch.nn.utils.parameters_to_vector(model.parameters()), torch.zeros_like(torch.nn.utils.parameters_to_vector(model.parameters())), reduction="sum")/(2*prior_sigma**2)
        else:
            regularization_loss = prior_logprob(torch.nn.utils.parameters_to_vector(model.parameters()))
        for i, (test_x, test_y) in enumerate(test_loader):
            model.eval()
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            test_pred = model(test_x)
            pred_loss = loss_fn(test_pred, test_y)/x.shape[0]  # we use sum-loss, and divide by the number of observations in batch
            loss = pred_loss + regularization_loss/total_obs_test # loss is now a per-observation loss
            epoch_test_loss += loss.item() / num_batches_test
            # pred_class = torch.argmax(test_pred, dim=-1)
            # current_correct_num += (pred_class == test_y).sum()
        # test_accuracy = current_correct_num.item() / total_obs_test
        # epoch_test_accuracy.append(test_accuracy)
        test_accuracy = 0
        epoch_test_losses.append(epoch_test_loss)
        test_loss_batch_number += [batch_number]
        txt = f"epoch = {epoch} \ttrain loss: {epoch_train_losses[-1]:2.5f}, train prediction loss: {epoch_pred_losses[-1]:2.5f}"+\
              f", regularization loss: {epoch_train_losses[-1]-epoch_pred_losses[-1]:2.5f}" + \
              f", norm of gradient: {epoch_train_grad[-1]:2.5f}, test loss: {epoch_test_losses[-1]:2.5f}" + \
              f", test accuracy: {test_accuracy*100:2.2f}, lr: {optimizer.param_groups[0]['lr']:4e}"
        if verbose and (epoch % print_every_epoch==0): logger_info(txt)
        if optimizer.param_groups[0]['lr']<1e-10:
            logger_info("Stopping training because lr is too low")
            break
    if plot:
        fig, ax = plt.subplots()
        ax.plot(train_loss_batch_number, all_train_losses, label="train loss")
        ax.plot(test_loss_batch_number, epoch_test_losses, label="test loss")
        ax.plot(test_loss_batch_number, epoch_train_losses, label="train avg over epoch")
        ax.set_xlim(0, batch_number)
        ax.set_ylim(min(all_train_losses), torch.tensor(all_train_losses).quantile(.99).item())
        ax.legend()
        fig.savefig(f"{plotpath}/loss.png")
        plt.close()

        plt.plot(train_loss_batch_number, step_train_grad, label="step gradient")
        plt.plot(test_loss_batch_number, epoch_train_grad, label="epoch gradient")
        plt.ylim(0, torch.tensor(step_train_grad).quantile(.99).item())
        plt.savefig(f"{plotpath}/gradient.png")
        plt.close()

        plt.plot(lrs)
        plt.savefig(f"{plotpath}/learning_rate.png")
        plt.close()

        plt.plot(test_loss_batch_number, epoch_train_accuracy, label="train accuracy")
        plt.plot(test_loss_batch_number, epoch_test_accuracy, label="test accuracy")
        plt.legend()
        plt.savefig(f"{plotpath}/accuracy.png")
        plt.close()

    return model, all_train_losses, lrs, epoch_train_losses, epoch_test_losses

