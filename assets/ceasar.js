'use strict';

// -----------------------------------------------------------------------------

var ceasar = ceasar || {};

// -----------------------------------------------------------------------------

ceasar.ENCRYPT = 1 << 0;
ceasar.DECRYPT = 1 << 1;

ceasar.cipher =
  function cipher(mode, text, key) {
    var alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-_.@';

    function mod(n, m) {
      return ((n % m) + m) % m;
    }

    function signSelector() {
      return mode === ceasar.ENCRYPT ? 1 : -1;
    }

    function newIndex(index) {
      return mod(index + signSelector() * key, alphabet.length);
    }

    return text
      .split('')
      .map(
        function newChar(char) {
          return alphabet.charAt(newIndex(alphabet.indexOf(char)));
        }
      )
      .join('');
  };

ceasar.encrypt = ceasar.cipher.bind(null, ceasar.ENCRYPT);

ceasar.decrypt = ceasar.cipher.bind(null, ceasar.DECRYPT);

// -----------------------------------------------------------------------------
