import numpy as np
from torchvision import datasets
from continuum.data_utils import create_task_composition, load_task_with_labels, shuffle_data
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns



class BreathDataset_v2(DatasetBase):
    def __init__(self, scenario, params, sequence_length= 20, overlap = 0.1, n_classes = 12):
        self.sequence_length = sequence_length
        self.n_classes = 12
        self.step  = int(sequence_length - overlap * sequence_length)
        dataset = 'breathing'
        self.download_load()
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks
        super(BreathDataset_v2, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)
    def download_load(self):
        trainset = np.load(file="./datasets/new_data_static/train_static.npy")[:, 1:]# for static dataset, there are 12 classes
        testset = np.load(file="./datasets/new_data_static/test_static.npy")[:, 1:]
        valset = np.load(file="./datasets/dynamic/dynamic_val_set.npy")[:, 1:]
        self.train_data, self.train_label = self.slice_window(data= trainset[:, :3], labels=trainset[:, -1])
        self.test_data, self.test_label = self.slice_window(data= testset[:, :3], labels=testset[:, -1])
        self.val_data, self.val_label = self.slice_window(data= valset[:, :3], labels=valset[:, -1])
        print("Generate Slice Window data successfully")
    def setup(self):
        if self.scenario == 'ni':
            self.train_set, self.val_set, self.test_set = construct_ns_multiple_wrapper(self.train_data,
                                                                                        self.train_label,
                                                                                        self.test_data, self.test_label,
                                                                                        self.task_nums, 32,
                                                                                        self.params.val_size,
                                                                                        self.params.ns_type, self.params.ns_factor,
                                                                                        plot=self.params.plot_sample)
        elif self.scenario == 'nc':
            self.task_labels = create_task_composition(class_nums=self.n_classes, num_tasks=self.task_nums, fixed_order=self.params.fix_order)
            self.test_set = []
            for labels in self.task_labels:
                x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
                self.test_set.append((x_test, y_test))
        else:
            raise Exception('wrong scenario')


    def new_task(self, cur_task, **kwargs):
        if self.scenario == 'ni':
            x_train, y_train = self.train_set[cur_task]
            labels = set(y_train)
        elif self.scenario == 'nc':
            labels = self.task_labels[cur_task]
            x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        return x_train, y_train, labels

        
    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def test_plot(self):
        test_ns(self.train_data[:10], self.train_label[:10], self.params.ns_type,
                                                         self.params.ns_factor)
    
    def dataset_info(self):
        return self.dataset

    def get_test_set(self):
        return self.test_set

    def clean_mem_test_set(self):
        self.test_set = None
        self.test_data = None
        self.test_label = None

    def slice_window(self, data, labels):
        X_local = []
        y_local = []
        for start in range(0, data.shape[0] - self.sequence_length, self.step):
            end = start + self.sequence_length
            X_local.append(data[start:end])
            y_local.append(labels[end-1])
        return np.array(X_local), np.array(y_local)