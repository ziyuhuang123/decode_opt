// SPDX-License-Identifier: MIT
//
// bench/json_min.h —— 极简 JSON 读取器（只读，够解析 models/models.json）。
// 不引入第三方依赖；解析失败时调用方回退到内置默认并打警告。
#pragma once

#include <cctype>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace jsonmin {

struct Value;
using ValuePtr = std::shared_ptr<Value>;

enum class Type { Null, Bool, Number, String, Array, Object };

struct Value {
  Type type = Type::Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<ValuePtr> arr;
  std::map<std::string, ValuePtr> obj;

  bool is_null() const { return type == Type::Null; }
  const Value* find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    auto it = obj.find(key);
    return it == obj.end() ? nullptr : it->second.get();
  }
  const Value* path(const std::vector<std::string>& keys) const {
    const Value* cur = this;
    for (const auto& k : keys) {
      if (!cur) return nullptr;
      cur = cur->find(k);
    }
    return cur;
  }
  double as_number(double dflt = 0.0) const {
    return type == Type::Number ? num : dflt;
  }
  int as_int(int dflt = 0) const {
    return type == Type::Number ? static_cast<int>(num) : dflt;
  }
  std::string as_string(const std::string& dflt = "") const {
    return type == Type::String ? str : dflt;
  }
  size_t size() const { return type == Type::Array ? arr.size() : 0; }
  const Value* at(size_t i) const {
    return (type == Type::Array && i < arr.size()) ? arr[i].get() : nullptr;
  }
};

class Parser {
 public:
  explicit Parser(const std::string& text) : s_(text) {}
  ValuePtr parse() {
    skip_ws();
    ValuePtr v = parse_value();
    return v;
  }
  const std::string& error() const { return err_; }

 private:
  const std::string& s_;
  size_t i_ = 0;
  std::string err_;

  void skip_ws() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
  }
  bool fail(const std::string& msg) {
    if (err_.empty()) {
      std::ostringstream os;
      os << msg << " at offset " << i_;
      err_ = os.str();
    }
    return false;
  }
  ValuePtr parse_value() {
    skip_ws();
    if (i_ >= s_.size()) { fail("unexpected end"); return nullptr; }
    const char c = s_[i_];
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') {
      auto v = std::make_shared<Value>();
      v->type = Type::String;
      if (!parse_string(v->str)) return nullptr;
      return v;
    }
    if (c == 't' || c == 'f') {
      auto v = std::make_shared<Value>();
      v->type = Type::Bool;
      if (s_.compare(i_, 4, "true") == 0) { v->b = true; i_ += 4; return v; }
      if (s_.compare(i_, 5, "false") == 0) { v->b = false; i_ += 5; return v; }
      fail("bad literal");
      return nullptr;
    }
    if (c == 'n') {
      if (s_.compare(i_, 4, "null") == 0) { i_ += 4; return std::make_shared<Value>(); }
      fail("bad literal");
      return nullptr;
    }
    auto v = std::make_shared<Value>();
    v->type = Type::Number;
    const size_t start = i_;
    if (c == '-' || c == '+') ++i_;
    while (i_ < s_.size() &&
           (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '.' ||
            s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '-' || s_[i_] == '+'))
      ++i_;
    if (start == i_) { fail("bad number"); return nullptr; }
    v->num = std::strtod(s_.substr(start, i_ - start).c_str(), nullptr);
    return v;
  }
  bool parse_string(std::string& out) {
    if (s_[i_] != '"') return fail("expected string");
    ++i_;
    out.clear();
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) return fail("bad escape");
        const char e = s_[i_++];
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case '/': out += '/'; break;
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case 'u': {
            if (i_ + 4 > s_.size()) return fail("bad \\u");
            const unsigned code = std::stoul(s_.substr(i_, 4), nullptr, 16);
            i_ += 4;
            if (code < 0x80) {
              out += static_cast<char>(code);
            } else if (code < 0x800) {
              out += static_cast<char>(0xC0 | (code >> 6));
              out += static_cast<char>(0x80 | (code & 0x3F));
            } else {
              out += static_cast<char>(0xE0 | (code >> 12));
              out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
              out += static_cast<char>(0x80 | (code & 0x3F));
            }
            break;
          }
          default: return fail("unknown escape");
        }
      } else {
        out += c;
      }
    }
    if (i_ >= s_.size()) return fail("unterminated string");
    ++i_;  // closing quote
    return true;
  }
  ValuePtr parse_array() {
    auto v = std::make_shared<Value>();
    v->type = Type::Array;
    ++i_;  // '['
    skip_ws();
    if (i_ < s_.size() && s_[i_] == ']') { ++i_; return v; }
    while (true) {
      ValuePtr e = parse_value();
      if (!e) return nullptr;
      v->arr.push_back(e);
      skip_ws();
      if (i_ < s_.size() && s_[i_] == ',') { ++i_; continue; }
      if (i_ < s_.size() && s_[i_] == ']') { ++i_; return v; }
      fail("expected , or ]");
      return nullptr;
    }
  }
  ValuePtr parse_object() {
    auto v = std::make_shared<Value>();
    v->type = Type::Object;
    ++i_;  // '{'
    skip_ws();
    if (i_ < s_.size() && s_[i_] == '}') { ++i_; return v; }
    while (true) {
      skip_ws();
      std::string key;
      if (!parse_string(key)) return nullptr;
      skip_ws();
      if (i_ >= s_.size() || s_[i_] != ':') { fail("expected :"); return nullptr; }
      ++i_;
      ValuePtr val = parse_value();
      if (!val) return nullptr;
      v->obj[key] = val;
      skip_ws();
      if (i_ < s_.size() && s_[i_] == ',') { ++i_; continue; }
      if (i_ < s_.size() && s_[i_] == '}') { ++i_; return v; }
      fail("expected , or }");
      return nullptr;
    }
  }
};

inline ValuePtr parse_file(const std::string& path, std::string* err) {
  std::ifstream in(path);
  if (!in) {
    if (err) *err = "cannot open " + path;
    return nullptr;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  Parser p(text);
  ValuePtr v = p.parse();
  if (!v && err) *err = p.error();
  return v;
}

}  // namespace jsonmin
