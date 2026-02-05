(function () {
  if (window.chmln) {
    return;
  }

  const queue = [];
  const chmln = function () {
    queue.push(arguments);
  };

  chmln.q = queue;
  chmln.enabled = false;
  window.chmln = chmln;
})();
