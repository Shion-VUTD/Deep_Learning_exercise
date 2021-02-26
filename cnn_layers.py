#CNNに使う関数とレイヤを実装していく
import numpy as np
from collections import OrderedDict
from neural_net2 import Relu,Affine,SoftmaxWithLoss

def im2col(input_data,filter_h,filter_w,stride=1,pad=0):
    #paddingこみ
    #画像データ(n,c,h,w)->２次元配列(n*oh*ow,c*fh*fw)に変換
    #必要なのは画像データ(n,c,h,w)とfh,fw
    n,c,h,w = input_data.shape
    out_h = (h+2*pad-filter_h)//stride +1
    out_w = (w+2*pad-filter_w)//stride +1

    #行列を格納する配列を準備
    col = np.zeros((n,c,filter_h,filter_w,out_h,out_w)) #3次元以上は配列で渡す
    #padding
    img = np.pad(input_data,[(0,0),(0,0),(pad,pad),(pad,pad)],'constant')

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            #変換
            col[:,:,y,x,:,:] = img[:,:,y:y_max:stride,x:x_max:stride]

    #colを２次元に変換
    col = col.transpose(0,4,5,1,2,3).reshape(n*out_h*out_w,-1)

    return col, out_h, out_w


def col2im(col,input_shape,filter_h,filter_w,stride=1,pad=0):
    #paddingこみ
    #画像データ(n,c,h,w)<-２次元配列(n*oh*ow,c*fh*fw)に変換
    #必要なのは2次元配列、変更したい画像データの形状、fhとfw

    n,c,h,w = input_shape
    out_h = (h+2*pad-filter_h)//stride +1
    out_w = (w+2*pad-filter_w)//stride +1

    #画像配列の準備
    img = np.zeros((n,c,h+2*pad,w+2*pad))   #-stride+1って何？
    #colを６次元に変形
    col = col.reshape(n,out_h,out_w,c,filter_h,filter_w).transpose(0,3,4,5,1,2)

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:,:,y,x,:,:] = img[:,:,y:y_max:stride,x:x_max:stride]

    #paddingは取って出力
    return img[:,:,pad:pad+h,pad:pad+w]      

#畳み込み層の実装
class Conv:
    #ハイパーパラメータは何？
    #フィルターの形状、枚数
    #strideとかpad

    def __init__(self,w,b,stride=1,pad=0):
        self.stride = stride
        self.pad = pad

        #学習のためにデータとフィルターを格納
        #データとフィルターは、生と２次元に直したやつと両方入れとく
        self.w = w
        self.col_w = None
        self.b = b
        self.x = None
        self.col = None
        self.dw = None
        self.db = None

    def forward(self,x):
        #2次元配列に
        self.x = x
        n,c,h,w = x.shape
        fn,c,fh,fw = self.w.shape
        col,out_h,out_w = im2col(x,filter_h=fh,filter_w=fw,stride=1,pad = 0)
        self.col = col
        self.col_w = self.w.reshape(-1,fn)

        out = np.dot(self.col,self.col_w)+self.b  #(n*oh*ow,fn)
        out = out.reshape((n,out_h,out_w,fn)).transpose(0,3,1,2)
        return out

    def backward(self,dout):   
        n,fn,out_h,out_w = dout.shape
        dout = dout.transpose(0,2,3,1).reshape(-1,fn)  #これで(n*oh*ow,fn)
        self.db = np.sum(dout,axis= 0)
        dcol = np.dot(dout,self.col_w.T)  #dcol.shape(n*oh*ow,c*fh*fw)
        dcol_w = np.dot(self.col.T,dout)  #dcol_w.shape = (c*fh*fw,fn)
        self.dw = dcol_w.transpose(1,0).reshape(self.w.shape)  #dw.shape=(fn,c,fh,fw)

        #dcol を画像データに変換
        dx = col2im(dcol,self.x.shape,self.w.shape[2],self.w.shape[3],stride = self.stride,pad=self.pad)

        return dx




class Pooling:
    #必要なもの
    #プーリングフィルターの形状
    #stride
    #pad
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        #逆伝播に必要なもの
        self.argmax = None   #各フィルターについて最大値をとるピクセルのインデックス配列
        self.input_shape = None

    def forward(self,x):
        n,c,h,w = x.shape
        self.input_shape = x.shape
        col,out_h,out_w = im2col(x,filter_h=self.pool_h,filter_w = self.pool_w,stride =self.stride, pad = self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)  #(n*oh*ow*c,fh*fw)
        self.argmax = np.argmax(col,axis = 1)
        Max = np.max(col,axis = 1)  #(n*oh*ow*c,)
        out = Max.reshape(n,out_h,out_w,c).transpose(0,3,1,2)

        return out

    def backward(self,dout):
        n,c,out_h,out_w = dout.shape
        dout = dout.transpose(0,2,3,1).flatten()

        dcol = np.zeros(n*out_h*out_w*c,self.pool_h*self.pool_w)
        dcol[np.arange(n*out_h*out_w*c),self.argmax] = dout

        #(忘れない！）dcolをcol2imに入る形で整形(n*out_h*out_w*c,self.pool_h*self.pool_w)->(n*out_h*out_w,c*self.pool_h*self.pool_w)
        dcol = dcol.reshape(n*out_h*out_w,-1)

        #dcolの整形
        dx = col2im(dcol,input_shape=self.input_shape,filter_h = self.pool_h,filter_w=self.pool_w,stride=self.stride,pad = self.pad)
        return dx

#１層目Conv,2層目Affine,3層目Affineの３層NN
class SimpleConvNet:
    #必要なもの
    # input_shape
    # １層目の出力層の形状を決定するハイパーパラメータ(filter_size,filter_num,stride,pad)
    #２層目の出力サイズhidden_size
    #3層目の出力サイズoutput_size
    #weight_init_std
    #activation_layers,use_batch_norm、use_dropout,weight_decay_lambda は今回は省略（ゼミやるときにはactivationとDropoutぐらいは実装して確認してみてもいいかも）
    
    def __init__(self,input_shape=(1,28,28),hidden_size=100,output_size = 10,Conv_param = {'filter_num':30,'filter_len':5,'stride':1,'pad':0},\
        pool_shape = (2,2),weight_init_std = 0.01):
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_init_std = weight_init_std
        #Conv_paramは、フィルターのの初期値に吸収させるため保持しなくてもOK
        
        filter_num = Conv_param['filter_num']
        filter_len = Conv_param['filter_size']
        self.pool_h = pool_shape[0]
        self.pool_w = pool_shape[1]
        self.stride = Conv_param['stride']
        self.pad = Conv_param['pad']

        #１層目出力層のサイズ
        output_len = (self.input_shape+2*self.pad-filter_len)//self.stride+1
        output_size = filter_num *output_len**2
        after_pool_size = filter_num * (output_len//self.pool_h) * (output_len//self.pool_w)

        #パラメータの初期値を格納
        self.params = {}
        self.params['w1'] = self.weight_init_std*np.random.randn((filter_num,input_shape[0],filter_len,filter_len))
        self.params['b1'] = np.zeros(filter_num)
        self.params['w2'] = self.weight_init_std*np.ramdom.randn(after_pool_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w3'] = self.weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b3'] = np.zeros(output_size)


        #レイヤの格納
        self.layers = OrderedDict()
        self.layers['Conv1'] = Conv(self.params['w1'],self.params['b1'],stride = self.stride,pad = self.pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pooling1'] = Pooling(pool_h = self.pool_h,pool_w=self.pool_w,stride = 1,pad = 0)
        self.layers['Affine1'] = Affine(self.params['w2'],self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['w3'],self.params['b3'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)    

            return x


    def loss(self,x,t):
        y = self.predict(x)
        loss = self.lastLayer.forward(y,t)
        return loss

    def gradient(self,x,t):
        self.loss(x,t)
        dout = 1
        dout = self.lastLayer.backward(dout)     
        
        #逆伝播
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #勾配の決定
        grads = {}
        grads['w1'] = self.layers['Conv1'].dw
        grads['b1'] = self.layers['Conv1'].db
        for i in (2,3):
            grads['w'+str(i)] = self.layers['Affine'+str(i-1)].dw
            grads['b'+str(i)] = self.layers['Affine'+str(i-1)].db

        return grads    

    #def accuracy(x,t):
    #====明日やる======








    
















        


            






        






        
            





      








