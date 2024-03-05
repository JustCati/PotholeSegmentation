import torch



def train_one_epoch_MASK(model, loader, optimizer, lr_scheduler, device):
    total_losses = []
    for images, targets in loader:
        images = list([image.to(device) for image in images])
        targets = [{k: v.to(device) for k, v in elem.items()} for elem in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        losses = {k: v.item() for k, v in loss_dict.items()}    #TODO: AGGIUNGI SCELTA SU METRICA DA OTTIMIZZARE
        losses["total_loss"] = loss.item()                      #TODO: RIMUOVI LA DIVISIONE PER IL NUMERO DI TARGETS

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # losses = {k: v / len(targets) for k, v in losses.items()}
        total_losses.append(losses)
    return {k: sum(loss[k] for loss in total_losses) / len(total_losses) for k in total_losses[0]}



def train_one_epoch_UNET(model, loss_func, loader, optimizer, lr_scheduler, device):
    total_losses = []
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [elem.to(device).unsqueeze(0) for elem in targets]

        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)

        pred = model(images)
        loss = loss_func(pred, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        total_losses.append(loss.item())
    return {"total_loss": sum(total_losses) / len(total_losses)}
