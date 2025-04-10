import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
device_ids = [0]
device = torch.device('cuda')


def load_stylegan2(ckpt_dir, channel_multiplier=2):
    '''
    <Input>
    ckpt_dir:      checkpoint direcotry    (str)    
    
    <Output>
    g_ema:         stylegan2 model
    '''
    
    print("\n>>> Loading StyleGAN...")
    
    from thirdparty.StyleCineGAN.models.stylegan2.model import Generator
    g_ema = Generator(1024, 512, 8, channel_multiplier=channel_multiplier)
    g_ckpt = torch.load(ckpt_dir)
    
    g_ema.load_state_dict(g_ckpt["g_ema"])
    g_ema = g_ema.eval().cuda()
    g_ema = nn.DataParallel(g_ema, device_ids=device_ids)
    
    print(">>> Loading done ------------------ \n")
    
    return g_ema


def load_encoder(ckpt_dir, encoder_type='fs', recon_idx=10):

    from argparse import Namespace

    print(">>> Loading StyleGAN encoder...")
    
    if encoder_type == 'fs':
        import os
        encoder_opts, encoder_config = set_encoder_args(os.getcwd(), ckpt_dir, recon_idx)
        trainer = load_fs_encoder(encoder_opts, encoder_config)
        print(">>> Loading done ------------------ \n")
        return trainer
    
    elif encoder_type == 'psp':
        
        ckpt = torch.load(ckpt_dir, map_location='cpu')
        opts = ckpt['opts']
        
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False

        opts['checkpoint_path'] = ckpt_dir
        opts = Namespace(**opts)

        from thirdparty.StyleCineGAN.models.encoders import pSp as encoder 
        net = encoder(opts)
        net.eval()
        print(">>> Loading done ------------------ \n")
        return net
    
    
def load_flownet(ckpt_dir, mode="sky"):
    
    print(">>> Loading flownet-pix2pixhd...")
    
    from thirdparty.StyleCineGAN.models.img2flow.pix2pixHD_model import InferenceModel
    sky_model = None
    fluid_model = None
    
    if (mode == "sky") or (mode == "sky+fluid"):
        sky_model = InferenceModel()
        opt = flownet_options(ckpt_dir, flownet_mode="sky")
        
        sky_model.initialize(opt)
        sky_model = sky_model.eval().cuda()
        sky_model = nn.DataParallel(sky_model, device_ids=device_ids)
        
    if (mode == "fluid") or (mode == "sky+fluid"):
        fluid_model = InferenceModel()
        opt = flownet_options(ckpt_dir, flownet_mode="fluid")
        
        fluid_model.initialize(opt)
        fluid_model = fluid_model.eval().cuda()
        fluid_model = nn.DataParallel(fluid_model, device_ids=device_ids)
        
    print(">>> Loading done ------------------ \n")
    
    return sky_model, fluid_model


def load_datasetgan(ckpt_dir, n_model=10, numpy_class=3):
    
    print(">>> Loading datasetGAN-mask...")
    
    device_ids = [0]
    from thirdparty.StyleCineGAN.models.DatasetGAN.classifier import pixel_classifier
    ckpt_dir = "./pretrained_models/datasetgan"
    
    classifier_list = []
    for i in range(n_model):
        classifier = pixel_classifier(numpy_class=numpy_class, dim=1472)
        classifier = nn.DataParallel(classifier, device_ids=device_ids).cuda()

        ckpt = torch.load(f"{ckpt_dir}/best_model_number_{i}.pth")
        classifier.load_state_dict(ckpt['model_state_dict'], strict=False)
        classifier.eval()
        
        classifier_list.append(classifier)
  
    print(">>> Loading done ------------------ \n")
    
    return classifier_list


# -----------------------------------------------------------------------------------------
def set_encoder_args(base_dir, pretrained_dir, idx_k):
    
    fs_dir = f"./thirdparty/StyleCineGAN/external_modules/feature_style_encoder"
    
    opts = {
        'config': '',
        'real_dataset_path': '',
        'dataset_path': '',
        'label_path': '',
        'stylegan_model_path': '',
        'w_mean_path': '',
        'arcface_model_path': '',
        'parsing_model_path': '',
        'log_path': '',
        'checkpoint': ''
    }
    
    opts['config'] = f"lhq_k{idx_k}"
    opts['log_path'] = f"{pretrained_dir}/logs/{opts['config']}"
    opts['stylegan_model_path'] = f"{pretrained_dir}/stylegan2-pytorch/sg2-lhq-1024.pt"
    opts['w_mean_path'] = f"{pretrained_dir}/stylegan2-pytorch/sg2-lhq-1024-mean.pt"
    opts['arcface_model_path'] = f"{pretrained_dir}/backbone.pth"
    opts['parsing_model_path'] = f"{pretrained_dir}/79999_iter.pth"
    
    from argparse import Namespace
    opts = Namespace(**opts)
    
    import yaml
    config = yaml.load(open(f'{fs_dir}/configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)
    opts.idx_k = config['idx_k']
    
    return opts, config


def load_fs_encoder(opts, config):
    
    import sys
    sys.path.append("./thirdparty/StyleCineGAN/external_modules/feature_style_encoder")
    from trainer import Trainer
    
    trainer = Trainer(config, opts)
    trainer.initialize(opts.stylegan_model_path, opts.arcface_model_path, opts.parsing_model_path, opts.w_mean_path)   
    trainer.to(device)
    
    trainer.load_model(opts.log_path)
    trainer.enc.eval()
    
    return trainer

        
# -----------------------------------------------------------------------------------------
def flownet_options(flownet_dir="./pretrained_models", flownet_mode="sky"):
    class ABC():
        pass

    opt = ABC()
    # opt = TestOptions().parse(save=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True #No shuffle
    opt.no_flip = True
    opt.no_instance = True
    opt.no_vgg = True
    opt.name = "img2flow"
    opt.checkpoints_dir = flownet_dir
    opt.which_epoch = 'latest'
    opt.verbose = True
    opt.gpu_ids = [0]
    opt.isTrain = False
    opt.resize_or_crop = "crop"
    opt.instance_feat = False  
    opt.label_feat = False
    opt.instance_feat = False
    opt.feat_num = 3
    opt.load_features = False
    opt.n_downsample_E = 4
    opt.nef = 16
    opt.n_clusters = 10

    opt.batchSize = 1
    opt.loadSize = 720
    opt.fineSize = 512
    opt.label_nc = 0
    opt.input_nc = 3
    opt.output_nc = 2

    opt.netG = 'global'
    opt.ngf = 64
    opt.n_downsample_global = 4
    opt.n_blocks_global = 9
    opt.n_blocks_local = 3
    opt.n_local_enhancers = 1
    opt.niter_fix_global = 0

    opt.norm = 'instance'
    opt.use_dropout = False
    opt.data_type = 32
    opt.fp16 = False
    opt.local_rank = 0
    if flownet_mode == "fluid":
        opt.name = "img2flow_fluid"
        
    elif flownet_mode == "sky":
        opt.name = "img2flow_sky"
    return opt