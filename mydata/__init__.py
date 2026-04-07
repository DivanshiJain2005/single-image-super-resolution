from importlib import import_module

from torch.utils.data.dataloader import default_collate  # 根据样本类型自动组合数据

from mydata.myDataLoader import MSDataLoader

class Data:
    def __init__(self, args):
        """ 直接构造loader并返回
        :param args: args
        """

        kwargs = {}
        if not args.cpu:  # 使用GPU进行计算
            kwargs['collate_fn'] = default_collate  # 将数据抽取出来放到一个batch中
            kwargs['pin_memory'] = True  # 是否将数据放在用于内存交换的内存中（page-locked)，从而加速GPU访问数据效率
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        '''导入trainSet，并构造trainLoader返回'''
        self.loader_train = None  # 用于存储加载过来的数据集
        # if not test only
        if not args.test_only:
            # 1.import_module动态导入对应的dataset模块
            module_train = import_module('mydata.' + args.data_train.lower())  # 之所以小写是因为mydata中的数据名称都是小写的
            # 2.生成trainset
            # 对应的getattr是在module_train的模块中获取参数为args.data_train的数据集的类，对应的(args)是这个函数的参数，
            # 用于参数初始化并返回数据集对象给trainset
            trainset = getattr(module_train, args.data_train)(args)
            # 3.load trainset
            self.loader_train = MSDataLoader(  # 训练数据加载器
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,  # 洗牌
                **kwargs  # 将其它的关键字参数包括num_work，pin_memory传递给父类的构造函数
            )

        '''导入testSet，并构造testLoader列表返回（支持多个测试数据集）'''
        benchmark_names = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']
        self.loader_test = []
        for dataset_name in args.data_test:
            # Temporarily override data_test for dataset construction
            args.data_test_single = dataset_name
            if dataset_name in benchmark_names:
                if not args.benchmark_noise:
                    module_test = import_module('mydata.benchmark')
                    testset = getattr(module_test, 'Benchmark')(args, train=False)
                else:
                    module_test = import_module('mydata.benchmark_noise')
                    testset = getattr(module_test, 'BenchmarkNoise')(args, train=False)
            else:
                module_test = import_module('mydata.' + dataset_name.lower())
                testset = getattr(module_test, dataset_name)(args, train=False)
            testset.dataset_name = dataset_name  # store name on dataset for trainer to read
            self.loader_test.append(
                MSDataLoader(args, testset, batch_size=1, shuffle=False, **kwargs)
            )
