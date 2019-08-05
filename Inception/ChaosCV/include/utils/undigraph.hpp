#pragma once

#include "core/core.hpp"

namespace chaos
{
	class CHAOS_API Undigraph
	{
	public:
		enum ConnectType
		{
			NONE,
			AVE,
			MAX,
		};

		Undigraph(int num);
		void Connect(int i, int j, float weight = 1.f, const ConnectType & type = NONE);

		std::vector<std::set<int>> Propagate(float th, float step, int max_size = 900);

	public:
		std::set<int> ConnectNodesConstraint(std::vector<std::set<int>>& groups, const std::set<int>& nodes, float th, int max_size);

		cv::SparseMat adjacent; // num x num
		int num;
	};
}