// 中英对照：每段左侧的 EN 按钮切换单段原文，页头按钮切换全部原文。
(function () {
  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (e) { return null; }
  }

  function init() {
    var blocks = document.querySelectorAll('[data-bi]');
    if (!blocks.length) return;

    blocks.forEach(function (el) {
      var id = el.getAttribute('data-bi');
      var en = document.querySelector('.en-orig[data-for="' + id + '"]');
      if (!en) return;
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'en-toggle';
      btn.textContent = 'EN';
      btn.setAttribute('aria-expanded', 'false');
      btn.setAttribute('aria-label', '显示这一段的英文原文');
      btn.addEventListener('click', function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        var open = !en.classList.contains('is-open');
        en.classList.toggle('is-open', open);
        el.classList.toggle('bi-open', open);
        btn.setAttribute('aria-expanded', String(open));
        btn.setAttribute('aria-label', open ? '收起这一段的英文原文' : '显示这一段的英文原文');
      });
      el.insertBefore(btn, el.firstChild);
    });

    // global switch in the page header
    var holder = document.querySelector('.content-icon-container');
    if (!holder) return;
    var all = document.createElement('button');
    all.type = 'button';
    all.className = 'bi-switch';
    all.innerHTML = '<span class="bi-dot" aria-hidden="true"></span>中英对照';
    holder.insertBefore(all, holder.firstChild);

    function set(on) {
      document.body.classList.toggle('bi-all', on);
      all.setAttribute('aria-pressed', String(on));
      all.title = on ? '关闭中英对照' : '每段下面显示英文原文';
    }
    var q = /[?&]en=1/.test(location.search);
    set(q || store('mj-bilingual') === '1');
    all.addEventListener('click', function () {
      var on = !document.body.classList.contains('bi-all');
      set(on);
      store('mj-bilingual', on ? '1' : '0');
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
