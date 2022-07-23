const data = {

  _cache: {},

  set: function (key, value) {
    this._cache[key] = value;
  },

  get: function (key) {
    return this._cache[key];
  },

  delete: function (key) {
    delete this._cache[key];
  }


};

module.exports = data;