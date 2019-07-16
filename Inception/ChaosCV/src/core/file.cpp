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


	void GetFileList(const std::string& folder, FileList& list, const std::string& types)
	{
		HANDLE handle;
		WIN32_FIND_DATA find_data;

		std::string root = folder;
		if (root.back() != '/' && root.back() != '\\')
			root.append("\\");

		static std::vector<std::string> type_list = Split(types, "\\|");

		handle = FindFirstFile((root + "*.*").c_str(), &find_data);
		if (handle != INVALID_HANDLE_VALUE)
		{
			do
			{
				if ('.' == find_data.cFileName[0])
				{
					continue;
				}
				else if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
				{
					GetFileList(root + find_data.cFileName, list, types);
				}
				else
				{
					std::string file_name = find_data.cFileName;

					size_t pos = file_name.find_last_of('.') + 1;
					std::string type = file_name.substr(pos);
					if ("*" == types || std::find(type_list.begin(), type_list.end(), type) != type_list.end())
					{
						list.push_back(root + file_name);
					}
				}
			} while (FindNextFile(handle, &find_data));
		}

		FindClose(handle);
	}
}