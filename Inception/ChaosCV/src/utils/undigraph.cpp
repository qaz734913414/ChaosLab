#include "utils/undigraph.hpp"

#include <stack>

namespace chaos
{
	Undigraph::Undigraph(int num) : num(num)
	{
		int shape[2] = {num, num};
		adjacent.create(2, shape, CV_32F);
	}
	void Undigraph::Connect(int i, int j, float weight, const ConnectType& type)
	{
		switch (type)
		{
		case AVE:
			if (adjacent.value<float>(i, j) != 0)
			{
				adjacent.ref<float>(i, j) = 0.5f * (adjacent.value<float>(i, j) + weight);
				adjacent.ref<float>(j, i) = 0.5f * (adjacent.value<float>(j, i) + weight);
			}
			else
			{
				adjacent.ref<float>(i, j) = weight;
				adjacent.ref<float>(j, i) = weight;
			}
			break;
		case MAX:
			adjacent.ref<float>(i, j) = std::max(adjacent.value<float>(i, j), weight);
			adjacent.ref<float>(j, i) = std::max(adjacent.value<float>(j, i), weight);
			break;
		case NONE:
		default:
			adjacent.ref<float>(i, j) = adjacent.ref<float>(j, i) = weight;
			break;
		}
	}

	std::vector<std::set<int>> Undigraph::Propagate(float th, float step, int max_size)
	{
		std::set<int> remain;
		for (int i = 0; i < num; i++)
			remain.insert(i);

		ProgressBar::Render("Clustering");
		std::vector<std::set<int>> group;
		// First iteration
		remain = ConnectNodesConstraint(group, remain, 0, max_size);
		ProgressBar::Update(group.size());
		// Iteration
		while (!remain.empty())
		{
			th = th + (1 - th) * step;
			remain = ConnectNodesConstraint(group, remain, th, max_size);
			ProgressBar::Update(group.size());
		}
		ProgressBar::Halt();

		return group;
	}

	std::set<int> Undigraph::ConnectNodesConstraint(std::vector<std::set<int>>& groups, const std::set<int>& nodes, float th, int max_size)
	{
		std::set<int> remain;
		std::set<int> removed;
		for (auto node : nodes)
		{
			std::set<int> conned;
			if (!removed.contains(node))
			{
				std::stack<int> stack;
				stack.push(node);
				while (!stack.empty())
				{
					int now = stack.top();
					conned.insert(now);
					removed.insert(now);
					stack.pop();

					// If num is very large, this searching speed will be very slow
					for (int i = 0; i < num; i++)
					{
						if (adjacent.value<float>(now, i) > th && !conned.contains(i))
						{
							stack.push(i);
						}
					}
				}

				std::vector<int> intersection;
				std::set_intersection(conned.begin(), conned.end(), remain.begin(), remain.end(), std::back_inserter(intersection));
				if (conned.size() > max_size || !intersection.empty())
				{
					//std::set_union(conned.begin(), conned.end(), remain.begin(), remain.end(), std::inserter(remain));
					// Set union
					for (auto n : conned) remain.insert(n);
				}
				else
				{
					groups.push_back(conned);
				}
			}
		}
		return remain;
	}

}