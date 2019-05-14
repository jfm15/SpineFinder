from keras import backend as K


def dice_coef_label(label):

    def dice_coef(y_true, y_pred, smooth=1):
        y_true_label = y_true[:, :, :, :, label]
        y_pred_label = y_pred[:, :, :, :, label]
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true_label * y_pred_label), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true_label),-1) + K.sum(K.square(y_pred_label), -1) + smooth)

    return dice_coef
