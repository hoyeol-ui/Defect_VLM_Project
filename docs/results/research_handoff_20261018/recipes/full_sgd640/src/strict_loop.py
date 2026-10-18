# Derived from pinned BaseTrainer._do_train. Only OOM/backend-error handler becomes bare raise.
def _do_train(self):
    """Perform the full training loop including setup, epoch iteration, validation, and final evaluation."""
    if self.world_size > 1:
        self._setup_ddp()
    self._setup_train()
    nb = len(self.train_loader)
    nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
    last_opt_step = -1
    self.epoch_time = None
    self.epoch_time_start = time.time()
    self.train_time_start = time.time()
    self.run_callbacks('on_train_start')
    LOGGER.info(f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\nUsing {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\nLogging results to {colorstr('bold', self.save_dir)}\nStarting training for " + (f'{self.args.time} hours...' if self.args.time else f'{self.epochs} epochs...'))
    if self.args.close_mosaic:
        base_idx = (self.epochs - self.args.close_mosaic) * nb
        self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    epoch = self.start_epoch
    self.optimizer.zero_grad()
    self._oom_retries = 0
    while True:
        self.epoch = epoch
        self.run_callbacks('on_train_epoch_start')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.scheduler.step()
        self._model_train()
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_loader)
        if epoch == self.epochs - self.args.close_mosaic:
            self._close_dataloader_mosaic()
            self.train_loader.reset()
        if RANK in {-1, 0}:
            LOGGER.info(self.progress_string())
            pbar = TQDM(enumerate(self.train_loader), total=nb)
        self.tloss = None
        for i, batch in pbar:
            self.run_callbacks('on_train_batch_start')
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]
                self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                for x in self.optimizer.param_groups:
                    x['lr'] = np.interp(ni, xi, [self.args.warmup_bias_lr if x.get('param_group') == 'bias' else 0.0, x['initial_lr'] * self.lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
            try:
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    if self.args.compile:
                        preds = self.model(batch['img'])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                self.scaler.scale(self.loss).backward()
            except RuntimeError as e:
                raise
            if ni - last_opt_step >= self.accumulate:
                self.optimizer_step()
                last_opt_step = ni
                if self.args.time:
                    self.stop = time.time() - self.train_time_start > self.args.time * 3600
                    if RANK != -1:
                        broadcast_list = [self.stop if RANK == 0 else None]
                        dist.broadcast_object_list(broadcast_list, 0)
                        self.stop = broadcast_list[0]
                    if self.stop:
                        break
            if RANK in {-1, 0}:
                loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                pbar.set_description(('%11s' * 2 + '%11.4g' * (2 + loss_length)) % (f'{epoch + 1}/{self.epochs}', f'{self._get_memory():.3g}G', *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)), batch.get('cls', batch['img']).shape[0], batch['img'].shape[-1]))
                self.run_callbacks('on_batch_end')
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)
            self.run_callbacks('on_train_batch_end')
            if self.stop:
                break
        else:
            self._oom_retries = 0
        if self._oom_retries and (not self.stop):
            continue
        if hasattr(unwrap_model(self.model).criterion, 'update'):
            unwrap_model(self.model).criterion.update()
        self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}
        self.run_callbacks('on_train_epoch_end')
        if RANK in {-1, 0}:
            self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
        final_epoch = epoch + 1 >= self.epochs
        if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
            self._clear_memory(None if self.device.type == 'mps' else 0.5)
            self.metrics, self.fitness = self.validate()
        if self._handle_nan_recovery(epoch):
            continue
        self.nan_recovery_attempts = 0
        if RANK in {-1, 0}:
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            if self.args.time:
                self.stop |= time.time() - self.train_time_start > self.args.time * 3600
            if (self.args.save or final_epoch) and self.save_model():
                self.run_callbacks('on_model_save')
        t = time.time()
        self.epoch_time = t - self.epoch_time_start
        self.epoch_time_start = t
        if self.args.time:
            mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
            self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
            self._setup_scheduler()
            self.scheduler.last_epoch = self.epoch
            self.stop |= epoch >= self.epochs
        self.run_callbacks('on_fit_epoch_end')
        self._clear_memory(None if self.device.type == 'mps' else 0.5)
        if RANK != -1:
            broadcast_list = [self.stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            self.stop = broadcast_list[0]
        if self.stop:
            break
        epoch += 1
    seconds = time.time() - self.train_time_start
    LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.')
    self.final_eval()
    if RANK in {-1, 0}:
        if self.args.plots:
            self.plot_metrics()
        self.run_callbacks('on_train_end')
    self._clear_memory()
    for loader in (self.train_loader, self.test_loader):
        if hasattr(loader, 'close'):
            loader.close()
    unset_deterministic()
    self.run_callbacks('teardown')
