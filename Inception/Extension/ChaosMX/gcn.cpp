#include "face/clusterer.hpp"
#include "utils/fast_search.hpp"
#include "utils/undigraph.hpp"
#include "utils/numpy.hpp"

namespace chaos
{
	namespace face
	{
		class GraphCN : public Clusterer
		{
		public:
			GraphCN(const dnn::Model& model, const dnn::Context& ctx)
			{
				int max_num_nodes = k_hops[0] * (k_hops[1] + 1) + 1;

				gcn = dnn::Net::Load(model, ctx);
				gcn->BindExecutor({ {"data0", {1, max_num_nodes, 512}}, {"data1", {1, max_num_nodes, max_num_nodes}} });

				searcher = FastSearcher::CreateFlat(512, FastSearcher::L2);
			}

			~GraphCN() final
			{

			}

			void Add(const dnn::Tensor& feat) final
			{
				feats.push_back(feat);
				searcher->Add(feat);
			}

			dnn::Tensor Cluster() final
			{
				int max_num_nodes = k_hops[0] * (k_hops[1] + 1) + 1;

				graph = Undigraph((int)feats.size());

				dnn::Tensor knn = dnn::Tensor({ (int)feats.size(), k_hops[0] + 1 }, S64);
				for (size_t i = 0; i < feats.size(); i++)
				{
					dnn::Tensor distance, labels;
					searcher->Search(feats[i], k_hops[0] + 1, distance, labels);

					memcpy((int64*)knn.data + i * (k_hops[0] + 1LL), labels.data, (k_hops[0] + 1LL) * sizeof(int64));
				}

				float th = FLT_MAX;
				for (int64 center_node = 0; center_node < feats.size(); center_node++)
				{
					Mat center_feat = feats[center_node];

					std::set<int64> unique_nodes = { center_node };
					std::set<int64> one_hop_nodes;
					int64* h0 = (int64*)knn.data + center_node * (k_hops[0] + 1LL);
					for (int i = 1; i < k_hops[0] + 1; i++)
					{
						unique_nodes.insert(h0[i]);
						one_hop_nodes.insert(h0[i]);
						int64* h1 = (int64*)knn.data + h0[i] * (k_hops[0] + 1LL);
						for (int j = 1; j < k_hops[1] + 1; j++)
						{
							unique_nodes.insert(h1[j]);
						}
					}

					std::map<int, int> unique_nodes_map; // unique_nodes_map
					int idx = 0;
					for (auto node : unique_nodes)
					{
						unique_nodes_map[node] = idx++;
					}

					//// one_hop_idcs
					//for (auto node : one_hop_nodes)
					//{
					//	std::cout << unique_nodes_map[node] << ", ";
					//}

					Mat data;
					for (auto i : unique_nodes)
					{
						Mat feat = feats[i];
						data.push_back(feat - center_feat);
					}
					cv::copyMakeBorder(data, data, 0, max_num_nodes - data.rows, 0, 0, cv::BORDER_CONSTANT);

					Mat A = Mat::zeros(unique_nodes.size(), unique_nodes.size(), CV_32F);
					for (auto i : unique_nodes)
					{
						int64* active = (int64*)knn.data + i * (k_hops[0] + 1LL);
						for (int j = 1; j < active_connection + 1; j++)
						{
							if (unique_nodes_map.find(active[j]) != unique_nodes_map.end())
							{
								A.at<float>(unique_nodes_map[i], unique_nodes_map[active[j]]) = 1;
								A.at<float>(unique_nodes_map[active[j]], unique_nodes_map[i]) = 1;
							}
						}
					}
					Mat D;
					cv::reduce(A, D, 1, cv::REDUCE_SUM);
					D = cv::repeat(D, 1, A.cols);
					A = A / D;
					cv::copyMakeBorder(A, A, 0, max_num_nodes - A.rows, 0, max_num_nodes - A.cols, cv::BORDER_CONSTANT);

					gcn->SetLayerData("data0", dnn::Tensor({ 1, max_num_nodes, 512 }, F32, data.data));
					gcn->SetLayerData("data1", dnn::Tensor({ 1, max_num_nodes, max_num_nodes }, F32, A.data));

					gcn->Forward();

					dnn::Tensor prob;
					gcn->GetLayerData("gcn0_dense1_sigmoid_fwd_output", prob);

					for (auto node : one_hop_nodes)
					{
						float score = ((float*)prob.data)[unique_nodes_map[node]];
						graph.Connect(center_node, node, score, Undigraph::AVE);
						th = th > score ? score : th;
					}
				}

				auto group = graph.Propagate(th, 0.6, 100);
				for (int i = 0; i < group.size(); i++)
				{
					if (group[i].size() > 1)
					{
						std::cout << "Group " << i << ":" << std::endl;
						for (auto node : group[i])
						{
							std::cout << node << " ";
						}
						std::cout << std::endl;
					}
				}

				return dnn::Tensor();
			}

		private:
			Ptr<dnn::Net> gcn;
			Ptr<FastSearcher> searcher;
			Undigraph graph;

			std::vector<dnn::Tensor> feats;

			int k_hops[2] = { 200, 5 };
			int active_connection = 5;
		};


		Ptr<Clusterer> Clusterer::LoadGCN(const dnn::Model& model, const dnn::Context& ctx)
		{
			return Ptr<Clusterer>(new GraphCN(model, ctx));
		}
	}
}