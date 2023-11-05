def CNNModel(num_layers: int, num_filters: [], filter_size: [], padding: []):
    assert len(num_filters) == num_layers
    assert len(filter_size) == num_layers
    assert len(padding) == num_layers
    x = (20 -filter_size[0] + 2 * padding[0]) + 1
    y = (20 -filter_size[0] + 2 * padding[0]) + 1
    x = x // 2  # Assuming max pool with kernel size 2
    y = y // 2
    cnn_model = nn.Sequential()
    cnn_model.add_module('conv_1', nn.Conv2d(in_channels=1,
                                             out_channels=num_filters[0],
                                             kernel_size=filter_size[0],
                                             padding=padding[0]))
    cnn_model.add_module('relu_1', nn.ReLU())
    cnn_model.add_module('maxpool_1', nn.MaxPool2d(kernel_size=2))
    for i in range(1, num_layers):
        cnn_model.add_module('conv_'+str(i), nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=filter_size[i], padding=padding[i]))
        cnn_model.add_module('relu_'+str(i), nn.ReLU())
        cnn_model.add_module('pool_' + str(i), nn.MaxPool2d(kernel_size=2))
        cnn_model.add_module('dropout' + str(i), nn.Dropout(p=0.5))
        x = (x - filter_size[i] + 2 * padding[i]) + 1  # Assuming stride=1
        y = (y - filter_size[i] + 2 * padding[i]) + 1
        x = x // 2  # Assuming max pool with kernel size 2
        y = y // 2
    cnn_model.add_module('flat', nn.Flatten())
    cnn_model.add_module('lin', nn.Linear(x*y*num_filters[-1], 3))
    return cnn_model