#pragma once

#include "face/face_info.hpp"
#include "dnn/net.hpp"

namespace chaos
{
	namespace face
	{
		class CHAOS_API Detector : public IndefiniteParameter
		{
		public:
			virtual std::vector<FaceInfo> Detect(const Mat& image) = 0;

			/// <summary>
			/// <para>Load Mutil-Task CNN models for face detection</para>
			/// <para>The models are trained by Matlab with caffe, so transpose is needed</para>
			/// <para>Implemented by MxNet</para>
			/// <para>Refer to "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks" </para>
			/// <para>Set Parameters:</para>
			/// <para>@ ScaleDecay: image pyramid scale decay, 0.709 for default</para>
			/// <para>@ MinFace: min face for detection, 40 for default</para>
			/// <para>@ NMS: nms threshold, 0.5 for default</para>
			/// <para>@ DoLandmark: return landmark if true, true for default</para>
			/// <para>@ Confidence: confidence vector for pnet, rnet and onet, [0.5,0.7,0.7] for default</para>
			/// </summary>
			/// <param name="folder">Models folder, include 3 models must be named PNet, RNet and ONet</param>
			/// <param name="ctx">Device type and id</param>
			static Ptr<Detector> LoadMTCNN(const std::string& folder, const dnn::Context& ctx = dnn::Context());
		};
	}
}