try:    
    from ThreeDLAPGAN.external.structural_losses.tf_nndistance import nn_distance
    from ThreeDLAPGAN.external.structural_losses.tf_approxmatch import approx_match, match_cost
except ImportError:
    print('Enable to load Chamfer-EMD')
except:
    print('External Losses (Chamfer-EMD) were not loaded.')
