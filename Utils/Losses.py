import torch.nn.functional as F


def d_lsgan(real_pred, fake_pred):
    real_loss = ((real_pred - 1.0) ** 2).mean()
    fake_loss = (fake_pred ** 2).mean()
    return 0.5 * (real_loss + fake_loss)


def g_lsgan(fake_pred):
    return ((fake_pred - 1.0) ** 2).mean()


def cycle_loss(original, reconstructed):
    return F.l1_loss(reconstructed, original)


def pixel_loss(original, translated):
    return F.l1_loss(translated, original)
