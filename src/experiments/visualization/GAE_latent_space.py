from sklearn.decomposition import PCA
import plotly.express as px

from src.processing.GAE_on_actors import get_latent_vector_generator
from src.utils.TM_dataset import actor_name, get_top_actors


# This file is supposed to show a dimensional reduction of visualization of actors latent vectors

def prepare_data():
    __plot_size = 100
    latent_vector_generator, actors_id = get_latent_vector_generator()
    data = []
    names = []
    for actor in get_top_actors(__plot_size):
        try:
            data.append(latent_vector_generator(actor))
            names.append(actor_name(actor)[0])
        except:
            pass
    return [data, names]


def draw_pca(data, names):
    pca = PCA(2)  # project from 128 to 2 dimensions
    projected = pca.fit_transform(data)
    fig = px.scatter(x=projected[:, 0], y=projected[:, 1], color=names)
    fig.show()


if __name__ == '__main__':
    data, names = prepare_data()
    draw_pca(data, names)
