#include "symbol.hpp"

namespace chaos
{
	namespace dnn
	{
		Symbol::Symbol() : op(""), name(""), attrs(std::map<std::string, std::string>()), inputs(std::vector<Inputs>()) {}
		Symbol::Symbol(Json json) : op(json.Data["op"]), name(json.Data["name"]), attrs(json["attrs"].Data)
		{
			auto nodes = json["inputs"];
			auto cnt = nodes.Data.size();
			for (size_t i = 0; i < cnt; i++)
			{
				auto node = nodes[i];
				Inputs in;
				in.node_id = std::atoi(node.Data["0"].c_str());
				in.index = std::atoi(node.Data["1"].c_str());
				in.version = std::atoi(node.Data["2"].c_str());

				inputs.push_back(in);
			}
		}

		SymbolList Load(const Json& json)
		{
			auto nodes = json["nodes"];

			size_t cnt = nodes.Data.size();

			SymbolList list;
			for (size_t i = 0; i < cnt; i++)
			{
				Symbol symbol = nodes[i];
				list.push_back(symbol);
			}

			return list;
		}
	}
}