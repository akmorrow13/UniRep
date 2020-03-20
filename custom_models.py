import tensorflow as tf

# Here I will try to implement a class which properly subclasses RNNCell
# but implements a mLSTM cell as above


class mLSTMCell(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self,
                 num_units,
                 weight_initializer=tf.compat.v1.orthogonal_initializer(),
                 bias_initializer=tf.compat.v1.constant_initializer(3),
                 wn_initializer=tf.compat.v1.ones_initializer(),
                 wn=True,
                 scope='mlstm',
                 var_device='cpu:0',
                 ):
        # Really not sure if I should reuse here
        super(mLSTMCell, self).__init__()
        self._num_units = num_units
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._wn_initializer = wn_initializer
        self._wn = wn
        self._scope = scope
        self._var_device = var_device

    @property
    def state_size(self):
        # The state is a tuple of c and h
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        # The output is h
        return (self._num_units)

    def zero_state(self, batch_size, dtype):
        c = tf.zeros([batch_size, self._num_units], dtype=dtype)
        h = tf.zeros([batch_size, self._num_units], dtype=dtype)
        return (c, h)

    def call(self, inputs, state):
        # Inputs will be a [batch_size, input_dim] tensor.
        # Eg, input_dim for a 10-D embedding is 10
        nin = inputs.get_shape()[1].value

        # Unpack the state tuple
        c_prev, h_prev = state
        with tf.compat.v1.variable_scope(self._scope):
            wx = tf.compat.v1.get_variable(
                "wx", [nin, self._num_units * 4], initializer=self._weight_initializer)
            wh = tf.compat.v1.get_variable(
                "wh", [self._num_units, self._num_units * 4], initializer=self._weight_initializer)
            wmx = tf.compat.v1.get_variable(
                "wmx", [nin, self._num_units], initializer=self._weight_initializer)
            wmh = tf.compat.v1.get_variable(
                "wmh", [self._num_units, self._num_units], initializer=self._weight_initializer)
            b = tf.compat.v1.get_variable(
                "b", [self._num_units * 4], initializer=self._bias_initializer)
            if self._wn:
                gx = tf.compat.v1.get_variable(
                    "gx", [self._num_units * 4], initializer=self._wn_initializer)
                gh = tf.compat.v1.get_variable(
                    "gh", [self._num_units * 4], initializer=self._wn_initializer)
                gmx = tf.compat.v1.get_variable(
                    "gmx", [self._num_units], initializer=self._wn_initializer)
                gmh = tf.compat.v1.get_variable(
                    "gmh", [self._num_units], initializer=self._wn_initializer)

        if self._wn:
            wx = tf.nn.l2_normalize(wx, axis=0) * gx
            wh = tf.nn.l2_normalize(wh, axis=0) * gh
            wmx = tf.nn.l2_normalize(wmx, axis=0) * gmx
            wmh = tf.nn.l2_normalize(wmh, axis=0) * gmh
        m = tf.matmul(inputs, wmx) * tf.matmul(h_prev, wmh)
        z = tf.matmul(inputs, wx) + tf.matmul(m, wh) + b
        i, f, o, u = tf.split(z, 4, 1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * c_prev + i * u
        h = o * tf.tanh(c)
        return h, (c, h)

class mLSTMCellStack(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self,
                 num_units,
                 num_layers=1,
                 dropout=None,
                 res_connect=True,
                 weight_initializer=tf.compat.v1.orthogonal_initializer(),
                 bias_initializer=tf.compat.v1.constant_initializer(3),
                 wn_initializer=tf.compat.v1.ones_initializer(),
                 wn=True,
                 scope='mlstm_stack',
                 var_device='cpu:0',
                 ):
        # Really not sure if I should reuse here
        super(mLSTMCellStack, self).__init__()
        self._num_units = num_units
        self._num_layers = num_layers
        self._dropout = dropout
        self._res_connect = res_connect
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._wn_initializer = wn_initializer
        self._wn = wn
        self._scope = scope
        self._var_device = var_device
        layers = [mLSTMCell(
                 num_units=self._num_units,
                 weight_initializer=self._weight_initializer,
                 bias_initializer=self._bias_initializer,
                 wn_initializer=self._wn_initializer,
                 wn=self._wn,
                 scope=self._scope + str(i),
                 var_device=self._var_device,
                 ) for i in range(self._num_layers)]
        if self._dropout:
            layers = [
                tf.contrib.rnn.DropoutWrapper(
                    layer, output_keep_prob=1-self._dropout) for layer in layers[:-1]] + layers[-1:]
        self._layers = layers

    @property
    def state_size(self):
        # The state is a tuple of c and h
        return (
            tuple(self._num_units for _ in range(self._num_layers)), 
            tuple(self._num_units for _ in range(self._num_layers))
            )

    @property
    def output_size(self):
        # The output is h
        return (self._num_units)

    def zero_state(self, batch_size, dtype):
        c_stack = tuple(tf.zeros([batch_size, self._num_units], dtype=dtype) for _ in range(self._num_layers))
        h_stack = tuple(tf.zeros([batch_size, self._num_units], dtype=dtype) for _ in range(self._num_layers))
        return (c_stack, h_stack)

    def call(self, inputs, state):
        # Inputs will be a [batch_size, input_dim] tensor.
        # Eg, input_dim for a 10-D embedding is 10

        # Unpack the state tuple
        c_prev, h_prev = state
        
        new_outputs = []
        new_cs = []
        new_hs = []
        for i, layer in enumerate(self._layers):
            if i == 0:
                h, (c,h_state) = layer(inputs, (c_prev[i],h_prev[i]))
            else:
                h, (c,h_state) = layer(new_outputs[-1], (c_prev[i],h_prev[i]))
            new_outputs.append(h)
            new_cs.append(c)
            new_hs.append(h_state)
        
        if self._res_connect:
            # Make sure number of layers does not affect the scale of the output
            scale_factor = tf.constant(1 / float(self._num_layers))
            final_output = tf.scalar_mul(scale_factor,tf.add_n(new_outputs))
        else:
            final_output = new_outputs[-1]

        return final_output, (tuple(new_cs), tuple(new_hs))


class myLSTMCell(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self,
                 num_units,
                 weight_initializer=tf.compat.v1.orthogonal_initializer(),
                 bias_initializer=tf.compat.v1.constant_initializer(3),
                 wn_initializer=tf.compat.v1.ones_initializer(),
                 wn=True,
                 scope='lstm',
                 var_device='cpu:0',
                 ):
        # Really not sure if I should reuse here
        super(myLSTMCell, self).__init__()
        self._num_units = num_units
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._wn_initializer = wn_initializer
        self._wn = wn
        self._scope = scope

    @property
    def state_size(self):
        # The state is a tuple of c and h
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        # The output is h
        return (self._num_units)

    def zero_state(self, batch_size, dtype):
        c = tf.zeros([batch_size, self._num_units], dtype=dtype)
        h = tf.zeros([batch_size, self._num_units], dtype=dtype)
        return (c, h)

    def call(self, inputs, state):
        # Inputs will be a [batch_size, input_dim] tensor.
        # Eg, input_dim for a 10-D embedding is 10
        nin = inputs.get_shape()[1].value

        # Unpack the state tuple
        c_prev, h_prev = state
        with tf.compat.v1.variable_scope(self._scope):
            # Weights from input to hidden layer
            wxg = tf.compat.v1.get_variable(
                "wxg", [nin, self._num_units], initializer=self._weight_initializer)
            wxi = tf.compat.v1.get_variable(
                "wxi", [nin, self._num_units], initializer=self._weight_initializer)
            wxf = tf.compat.v1.get_variable(
                "wxf", [nin, self._num_units], initializer=self._weight_initializer)
            wxo = tf.compat.v1.get_variable(
                "wxo", [nin, self._num_units], initializer=self._weight_initializer)
            # Weights from hidden (-1) to hidden layer
            whg = tf.compat.v1.get_variable(
                "whg", [self._num_units, self._num_units], initializer=self._weight_initializer)
            whi = tf.compat.v1.get_variable(
                "whi", [self._num_units, self._num_units], initializer=self._weight_initializer)
            whf = tf.compat.v1.get_variable(
                "whf", [self._num_units, self._num_units], initializer=self._weight_initializer)
            who = tf.compat.v1.get_variable(
                "who", [self._num_units, self._num_units], initializer=self._weight_initializer)
            
            # Biases
            bg = tf.compat.v1.get_variable(
                "bg", [self._num_units], initializer=tf.compat.v1.constant_initializer(0))
            bi = tf.compat.v1.get_variable(
                "bi", [self._num_units], initializer=tf.compat.v1.constant_initializer(0))
            # Forget bias should be nonzero
            bf = tf.compat.v1.get_variable(
                "bf", [self._num_units], initializer=self._bias_initializer)
            bo = tf.compat.v1.get_variable(
                "bo", [self._num_units], initializer=tf.compat.v1.constant_initializer(0))

            if self._wn:
                gxg = tf.compat.v1.get_variable(
                    "gxg", [self._num_units], initializer=self._wn_initializer)
                gxi = tf.compat.v1.get_variable(
                    "gxi", [self._num_units], initializer=self._wn_initializer)
                gxf = tf.compat.v1.get_variable(
                    "gxf", [self._num_units], initializer=self._wn_initializer)
                gxo = tf.compat.v1.get_variable(
                    "gxo", [self._num_units], initializer=self._wn_initializer)
                ghg = tf.compat.v1.get_variable(
                    "ghg", [self._num_units], initializer=self._wn_initializer)
                ghi = tf.compat.v1.get_variable(
                    "ghi", [self._num_units], initializer=self._wn_initializer)
                ghf = tf.compat.v1.get_variable(
                    "ghf", [self._num_units], initializer=self._wn_initializer)
                gho = tf.compat.v1.get_variable(
                    "gho", [self._num_units], initializer=self._wn_initializer)

        if self._wn:
            wxg = tf.nn.l2_normalize(wxg, axis=0) * gxg
            wxi = tf.nn.l2_normalize(wxi, axis=0) * gxi
            wxf = tf.nn.l2_normalize(wxf, axis=0) * gxf
            wxo = tf.nn.l2_normalize(wxo, axis=0) * gxo
            whg = tf.nn.l2_normalize(whg, axis=0) * ghg
            whi = tf.nn.l2_normalize(whi, axis=0) * ghi
            whf = tf.nn.l2_normalize(whf, axis=0) * ghf
            who = tf.nn.l2_normalize(who, axis=0) * gho

        g = tf.nn.tanh(tf.matmul(inputs, wxg) + tf.matmul(h_prev, whg) + bg)
        i = tf.nn.sigmoid(tf.matmul(inputs, wxi) + tf.matmul(h_prev, whi) + bi)
        f = tf.nn.sigmoid(tf.matmul(inputs, wxf) + tf.matmul(h_prev, whf) + bf)
        o = tf.nn.sigmoid(tf.matmul(inputs, wxo) + tf.matmul(h_prev, who) + bo)

        c = f * c_prev + i * g
        h = o * tf.nn.tanh(c)
        return h, (c, h) 

class myGRUCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """ 
    To keep the signature of the other LSTM classes, this will
    return a duplicate tuple of the hidden state and another hidden state
    """
    def __init__(self,
                 num_units,
                 weight_initializer=tf.compat.v1.orthogonal_initializer(),
                 bias_initializer=tf.compat.v1.constant_initializer(0),
                 wn_initializer=tf.compat.v1.ones_initializer(),
                 wn=True,
                 scope='gru',
                 var_device='cpu:0',
                 ):
        # Really not sure if I should reuse here
        super(myGRUCell, self).__init__()
        self._num_units = num_units
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._wn_initializer = wn_initializer
        self._wn = wn
        self._scope = scope

    @property
    def state_size(self):
        # The state is a tuple of h and h (duplicated- see docstring)
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        # The output is h
        return (self._num_units)

    def zero_state(self, batch_size, dtype):
        h = tf.zeros([batch_size, self._num_units], dtype=dtype)
        h_dup = tf.zeros([batch_size, self._num_units], dtype=dtype)
        return (h, h_dup)

    def call(self, inputs, state):
        # Inputs will be a [batch_size, input_dim] tensor.
        # Eg, input_dim for a 10-D embedding is 10
        nin = inputs.get_shape()[1].value

        # Unpack the state tuple
        h_prev, __ = state
        with tf.compat.v1.variable_scope(self._scope):
            # Weights from input to hidden layer
            wxz = tf.compat.v1.get_variable(
                "wxf", [nin, self._num_units], initializer=self._weight_initializer)
            wxr = tf.compat.v1.get_variable(
                "wxr", [nin, self._num_units], initializer=self._weight_initializer)
            wxh = tf.compat.v1.get_variable(
                "wxh", [nin, self._num_units], initializer=self._weight_initializer)

            # Weights from hidden (-1) to hidden layer
            whz = tf.compat.v1.get_variable(
                "whg", [self._num_units, self._num_units], initializer=self._weight_initializer)
            whr = tf.compat.v1.get_variable(
                "whi", [self._num_units, self._num_units], initializer=self._weight_initializer)
            whh = tf.compat.v1.get_variable(
                "whf", [self._num_units, self._num_units], initializer=self._weight_initializer)

            
            # Biases
            bz = tf.compat.v1.get_variable(
                "bz", [self._num_units], initializer=tf.compat.v1.constant_initializer(0))
            br = tf.compat.v1.get_variable(
                "br", [self._num_units], initializer=tf.compat.v1.constant_initializer(0))
            bh = tf.compat.v1.get_variable(
                "bh", [self._num_units], initializer=self._bias_initializer)

            if self._wn:
                gxz = tf.compat.v1.get_variable(
                    "gxz", [self._num_units], initializer=self._wn_initializer)
                gxr = tf.compat.v1.get_variable(
                    "gxr", [self._num_units], initializer=self._wn_initializer)
                gxh = tf.compat.v1.get_variable(
                    "gxh", [self._num_units], initializer=self._wn_initializer)
                ghz = tf.compat.v1.get_variable(
                    "ghz", [self._num_units], initializer=self._wn_initializer)
                ghr = tf.compat.v1.get_variable(
                    "ghr", [self._num_units], initializer=self._wn_initializer)
                ghh = tf.compat.v1.get_variable(
                    "ghh", [self._num_units], initializer=self._wn_initializer)


        if self._wn:
            wxz = tf.nn.l2_normalize(wxz, axis=0) * gxz
            wxr = tf.nn.l2_normalize(wxr, axis=0) * gxr
            wxh = tf.nn.l2_normalize(wxh, axis=0) * gxh
            whz = tf.nn.l2_normalize(whz, axis=0) * ghz
            whr = tf.nn.l2_normalize(whr, axis=0) * ghr
            whh = tf.nn.l2_normalize(whh, axis=0) * ghh


        z = tf.nn.sigmoid(tf.matmul(inputs, wxz) + tf.matmul(h_prev, whz) + bz)
        r = tf.nn.sigmoid(tf.matmul(inputs, wxr) + tf.matmul(h_prev, whr) + br)
        g = tf.nn.tanh(tf.matmul(inputs, wxh) + tf.matmul(r * h_prev, whh) + bh)
        h = z * h_prev + (1 - z) * g

        return h, (h, h) 
