import torch
from framework import sc_net_framework
from tqdm import tqdm
from config import opt as opt1

def train(num_epochs=200, lr=1e-4, device='cuda:0', save_path='model_58x40x8'):

    fw = sc_net_framework(pattern='fine_tuning', cfg=opt1)
    model = fw.model.to(device)
    loss_fn = fw.loss_fn.to(device)
    train_loader = fw.dataLoader_train
    eval_loader = fw.dataLoader_eval

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_val_loss = float('inf')
    
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(eval_loader)}")
    print(f"Starting training for {num_epochs} epochs\n")

    for epoch in range(num_epochs):

        # --- training ---
        model.train()
        model.pattern = 'training'
        model.sampling_point_framework.pattern = 'training'
        model.object_detection_framework.pattern = 'training'

        train_loss = 0.0
        train_bar = tqdm(train_loader, 
                         desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                         leave=False)
        for images, targets in train_bar:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            od_outputs, sc_outputs = model(images)
            loss = loss_fn(od_outputs, sc_outputs, targets)

            # add after od_outputs, sc_outputs = model(images) in training loop
            if epoch == 0 and train_loss == 0.0:  # first batch only
                with torch.no_grad():
                    logits = od_outputs['pred_logits']   # (B, 16, num_classes+1)
                    boxes  = od_outputs['pred_boxes']    # (B, 16, 2)
                    probs  = torch.softmax(logits, dim=-1)
                    bg_prob    = probs[:, :, -1].mean().item()   # avg background prob
                    max_lesion = probs[:, :, :-1].max().item()   # highest lesion prob anywhere
                    print(f"\n[DIAG epoch {epoch+1}] "
                        f"avg_bg_prob={bg_prob:.4f} "
                        f"max_lesion_prob={max_lesion:.4f} "
                        f"gt_boxes={sum(len(t['boxes']) for t in targets)}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss /= len(train_loader)

        # --- validation ---
        model.train()
        model.pattern = 'training'
        model.sampling_point_framework.pattern = 'training'
        model.object_detection_framework.pattern = 'training'

        val_loss = 0.0
        val_bar = tqdm(eval_loader,
                       desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ",
                       leave=False)
        with torch.no_grad():
            for images, targets in val_bar:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                od_outputs, sc_outputs = model(images)
                loss = loss_fn(od_outputs, sc_outputs, targets)
                val_loss += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss /= len(eval_loader)
        
        # save every epoch
        epoch_path = f"{save_path}_epoch{epoch+1:03d}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, epoch_path)

        marker = " *" if val_loss < best_val_loss else ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_path}_best.pth")

        print(f"Epoch {epoch+1:03d}/{num_epochs} | "
              f"train: {train_loss:.4f} | "
              f"val: {val_loss:.4f} | "
              f"lr: {scheduler.get_last_lr()[0]:.2e} | "
              f"saved: {epoch_path}"
              f"{marker}")

if __name__ == '__main__':
    train(device='cuda:0', save_path='DEBUG_RUN')