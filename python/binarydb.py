import os
import caffe.proto.caffe_pb2 as cpb
import google.protobuf as pb
from google.protobuf import text_format

class BinaryDB:

    class SampleProps:
        pass

    def __init__(self, param_str, num_top_blobs):
        # Get the LayerParameters
        param = cpb.LayerParameter()
        param.ParseFromString(param_str)
        print(param)

        self.data_param = param.data_param

        # Read clip list
        assert(os.path.isfile(self.data_param.clip_list))

        with open(self.data_param.clip_list) as f:
            self.clips = [line.rstrip('\n') for line in f if len(line.rstrip('\n')) > 0]

        print("Clips: ")
        print(self.clips)

        # Set up sample offset ranges
        self.sample_props = []
        self.setUpSamples()

        # Find valid samples in all clips
        self.findSamplesInClips()

    def setUpSamples(self):

        for sample in self.data_param.sample:
            min_offset = None
            max_offset = None

            for entry in sample.entry:
                if min_offset is None or min_offset > entry.offset:
                    min_offset = entry.offset
                if max_offset is None or max_offset < entry.offset:
                    max_offset = entry.offset

            self.sample_props.append(BinaryDB.SampleProps)
            self.sample_props[-1].min_offset = min_offset
            self.sample_props[-1].max_offset = max_offset

    def findSamplesInClips(self):
        for clip_folder in self.clips:
            index_file = os.path.join(self.data_param.source, clip_folder, "index.prototxt")
            assert(os.path.isfile(index_file))
            with open(index_file) as f:
                index_str = f.read()

            print("File: " + index_file)
            bindb_index = cpb.BinaryDB()
            text_format.Merge(index_str, bindb_index)
            print(bindb_index)


    def getInfos(self):
        print("Called getInfos")
        return []
    