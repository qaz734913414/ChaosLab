#include "core/core.hpp"

#include <regex>

#include <Windows.h>


namespace chaos
{
	File::File() {}
	File::File(const std::string& _file)
	{
		if (_file.empty())
			return;

		std::string file = _file;
		for (auto& f : file)
		{
			f = f == '/' ? '\\' : f;
		}

		auto spos = file.find_last_of('\\') + 1;
		path = file.substr(0, spos);

		file = file.substr(spos);
		// ppos == 0 means that the string does not contain '.' 
		auto ppos = file.find_last_of('.') + 1;

		name = 0 == ppos ? file : file.substr(0, ppos - 1);
		type = 0 == ppos ? "" : file.substr(ppos);

		// if the first char is '.', the name will be empty
		//CHECK(!name.empty());
		auto valid = std::regex_match(name, std::regex("[^\\|\\\\/:\\*\\?\"<>]+"));
		//CHECK(valid) << "File name can not contain |\\/:*?\"<>";
	}
	File::File(const std::string& _path, const std::string& _name, const std::string& _type) : path(_path), name(_name), type(_type)
	{
		for (auto& p : path)
		{
			p = p == '/' ? '\\' : p;
		}

		if (path.back() != '\\')
			path.append("\\");

		//CHECK(!name.empty());
		auto valid = std::regex_match(name, std::regex("[^\\|\\\\/:\\*\\?\"<>]+"));
		//CHECK(valid) << "File name can not contain |\\/:*?\"<>";
	}

	File::operator std::string() const
	{
		return name.empty() ? std::string() : path + name + "." + type;
	}

	std::ostream& operator << (std::ostream& stream, const File& file)
	{
		return stream << std::string(file);
	}

	std::string File::GetName() const
	{
		return name;
	}
	std::string File::GetPath() const
	{
		return path;
	}
	std::string File::GetType() const
	{
		return type;
	}
}