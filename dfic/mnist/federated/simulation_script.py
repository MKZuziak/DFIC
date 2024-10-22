import os
from functools import partial

from torch import optim

import timm
import pickle

from FedJust.model.federated_model import FederatedModel
from FedJust.node.federated_node import FederatedNode
from FedJust.simulation.adaptive_optimizer_simulation import Adaptive_Optimizer_Simulation
from FedJust.aggregators.fedopt_aggregator import Fedopt_Optimizer
from FedJust.files.archive import create_archive

def simulation():
    (metrics_savepath, 
     nodes_models_savepath, 
     orchestrator_model_savepath) = create_archive(os.getcwd())

    dataset_path = r'/home/maciejzuziak/raid/DL_course/DFIC/dfic/mnist/federated/dataset/MNIST_10_dataset_pointers'
    with open(dataset_path, 'rb') as file:
        data = pickle.load(file)
    orchestrator_data = data[0]
    nodes_data = data[1]
    net_architecture = timm.create_model('resnet18', num_classes=10, in_chans=1, pretrained=False)
    optimizer_architecture = partial(optim.SGD, lr=0.001, momentum=0.9)
    model_tempate = FederatedModel(
        net=net_architecture,
        optimizer_template=optimizer_architecture,
        loader_batch_size=32
    )
    node_template = FederatedNode()
    fed_avg_aggregator = Fedopt_Optimizer()

    simulation_instace = Adaptive_Optimizer_Simulation(model_template=model_tempate,
                                    node_template=node_template)
    simulation_instace.attach_orchestrator_model(orchestrator_data=orchestrator_data)
    simulation_instace.attach_node_model({
        node: nodes_data[node] for node in range(10)
    })
    simulation_instace.training_protocol(
        iterations=50,
        sample_size=2,
        local_epochs=2,
        aggrgator=fed_avg_aggregator,
        learning_rate=1.0,
        metrics_savepath=metrics_savepath,
        nodes_models_savepath=nodes_models_savepath,
        orchestrator_models_savepath=orchestrator_model_savepath
    )
    

if __name__ == "__main__":
    simulation()