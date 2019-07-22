#include "symbol.hpp"

namespace chaos
{
	namespace dnn
	{
		Inputs::Inputs()
		{
			node_id = 0;
			index = 0;
			version = 0;
		};
		Inputs::Inputs(const Json& json)
		{
			node_id = std::stoi(json.Data["0"]);
			index = std::stoi(json.Data["1"]);
			version = std::stoi(json.Data["2"]);
		}

		Symbol::Symbol() : op(Operator()), name(""), attrs(Attrs()), inputs(std::vector<Inputs>()) {}
		Symbol::Symbol(const Json& json)
		{
			op = Operator(json.Data["op"]);
			name = json.Data["name"];
			attrs = json["attrs"].Data;

			auto input = json["inputs"];
			auto cnt = input.Data.size();
			for (size_t i = 0; i < cnt; i++)
			{
				inputs.push_back(input[i]);
			}
		}
	}
}