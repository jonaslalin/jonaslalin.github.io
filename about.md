---
layout: page
title: About
---

__Hello!__

My name is Jonas Lalin. I am a deep learning enthusiast from Stockholm, Sweden, and I love math. Here you can read about the mechanics of deep learning. I hope you enjoy and learn something new from reading my posts! 🧙

Would you like to get in touch? Pass my completely automated public Turing test to tell computers and humans apart by subtracting the $$6$$th prime number from

> the answer to the Ultimate Question of Life, the Universe, and Everything.

<form id="captcha">
  <label for="key">Key:</label>
  <input type="number" id="key" name="key" required min="0" value="13">
  <button type="submit">Decipher</button>
</form>
Plaintext: `z43q8o1q1y3pw2qy1os42`{: #plaintext }

<script src="{% link /assets/ceasar.js %}"></script>

<script>
  var ciphertext = '.dc3h1a3a_c29b3_a15db';
  var key = document.getElementById('key');
  var plaintext = document.getElementById('plaintext');

  var form = document.getElementById('captcha');
  form.addEventListener('submit', function onSubmit(event) {
    plaintext.textContent = ceasar.decrypt(ciphertext, key.value);
    event.preventDefault();
  });
</script>
