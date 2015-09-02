import caffe
import caffe.proto.caffe_pb2 as cpb
import google.protobuf.text_format


class BinaryDB:

    def __init__(self, param_str, num_top_blobs):
        param = cpb.LayerParameter()
        param.ParseFromString(param_str)

        print(param)



    def getInfos(self):
        print("Called getInfos")
        return []
    