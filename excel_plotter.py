import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

xls = pd.ExcelFile('Generative Classifier Experiments.xlsx')
sheets = ['L2 FGM Attack', 'L-inf FGM Attack', 'L2 PGD Attack',
          'L-inf PGD Attack', 'L2 CW Attack', 'Additive Uniform Noise Attack',
          'Additive Gaussian Noise Attack', 'Linear Search Contrast Reductio',
          'Binary Search Contrast Reductio', 'Gaussian Blur Attack']

# just to fix chopped-off sheet names
attack_names = {
    'L2 FGM Attack': 'L2 FGM Attack',
    'L-inf FGM Attack': 'L-inf FGM Attack',
    'L2 PGD Attack': 'L2 PGD Attack',
    'L-inf PGD Attack': 'L-inf PGD Attack',
    'L2 CW Attack': 'L2 CW Attack',
    'Additive Uniform Noise Attack': 'Additive Uniform Noise Attack',
    'Additive Gaussian Noise Attack': 'Additive Gaussian Noise Attack',
    'Linear Search Contrast Reductio': 'Linear Search Contrast Reduction',
    'Binary Search Contrast Reductio': 'Binary Search Contrast Reduction',
    'Gaussian Blur Attack' : 'Gaussian Blur Attack'
}

discriminative_models = {
    0: "Logistic Regression",
    2: "Neural Network",
    3: "Convolutional NN",
}
generative_models = {
    5: "GaussianNB",
    7: "GMM",
    9: "Neural Network VAE",
    10: "CNN VAE",
    11: "CNN VAE Binary",
    12: "CNN VAE + CRF",
    13: "MRF",
    14: "VAE + MRF"
}

# define this based on the relevant indices, i.e. number of epsilons (including the base value 0)
start_idx = 2
end_idx = 13

fontP = FontProperties()
fontP.set_size('x-small')

# plot across the attack dimension, i.e. how different models perform under a particular attack
for sheet in attack_names.keys():
    df = pd.read_excel(xls, sheet)
    # fix first epsilon
    eps = df.columns[start_idx:end_idx].values
    eps[0] = 0


    # plot for discriminative models
    for idx, dm in discriminative_models.items():
        acc = df.loc[idx][start_idx:end_idx]
        plt.plot(eps, acc, label=dm, marker='o')
    fig_title = "Discriminative Models Against " + attack_names[sheet]
    plt.title(fig_title)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', prop=fontP)
    plt.tight_layout()
    plt.show()
    # plt.savefig(fig_title + ".png")

    # plot for generative models
    for idx, gm in generative_models.items():
        acc = df.loc[idx][start_idx:end_idx].values
        plt.plot(eps, acc, label=gm, marker='o')
    fig_title = "Generative Models Against " + attack_names[sheet]
    plt.title(fig_title)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', prop=fontP)
    plt.tight_layout()
    plt.show()
    # plt.savefig(fig_title + ".png")

# plot across the model dimension, i.e. how a particular model perform under different attacks
for idx, dm in discriminative_models.items():
    for sheet in attack_names.keys():
        df = pd.read_excel(xls, sheet)
        acc = df.loc[idx][start_idx:end_idx].values
        eps = df.columns[start_idx:end_idx].values
        eps[0] = 0
        plt.plot(eps, acc, label=attack_names[sheet], marker='o')
    plt.title(dm)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', prop=fontP)
    plt.tight_layout()
    plt.show()
    # plt.savefig(dm + ".png")


for idx, gm in generative_models.items():
    for sheet in attack_names.keys():
        df = pd.read_excel(xls, sheet)
        acc = df.loc[idx][start_idx:end_idx].values
        eps = df.columns[start_idx:end_idx].values
        eps[0] = 0
        plt.plot(eps, acc, label=attack_names[sheet], marker='o')
    plt.title(gm)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', prop=fontP)
    plt.tight_layout()
    plt.show()
    # plt.savefig(gm + ".png")
