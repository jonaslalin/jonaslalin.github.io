// -----------------------------------------------------------------------------

function ceasarCipher(mode, text, key) {
  var alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-_.@';

  function mod(n, m) {
    return ((n % m) + m) % m;
  }

  function signSelector() {
    return mode == 'encrypt' ? 1 : -1;
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
}

var encrypt = ceasarCipher.bind(null, 'encrypt');

var decrypt = ceasarCipher.bind(null, 'decrypt');

// -----------------------------------------------------------------------------

var plaintext = document.getElementById('plaintext');
var ciphertext = '.dc3h1a3a_c29b3_a15db';
var key = document.getElementById('key');

var form = document.getElementById('captcha');
form.addEventListener('submit', function onSubmit(event) {
  plaintext.textContent = decrypt(ciphertext, key.value);
  event.preventDefault();
});

// -----------------------------------------------------------------------------
