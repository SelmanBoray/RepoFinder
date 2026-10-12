# RepoFinder
*[English](README.md)*

RepoFinder, büyük bir açık kaynak projeye katkıda bulunmadan önce o projede
yolunuzu bulmanıza yardım eder. Ona bir GitHub deposu verdiğinizde bir mimari
haritası çizer ve "X özelliği nerede uygulanmış?" sorusunu koda işaret eden
`path:start-end` alıntılarıyla yanıtlar.

## Kurulum

Python 3.10+ ve `PATH` üzerinde `git` gerekir.

```powershell
git clone <this repository> RepoFinder
cd RepoFinder
python -m venv .venv
.venv\Scripts\python -m pip install -e .
```

macOS / Linux'ta `.venv/bin/python` kullanın. Kurulum, sanal ortama bir
`repofinder` komutu ekler (Windows'ta `.venv\Scripts\repofinder.exe`); ortamı
etkinleştirin ya da komutu bu yolla çağırın.

## LLM erişimi: iki seçenek

`index`, `map` ve `search` çevrimdışı çalışır (Jev kurulmuşsa `search`, yeniden sıralamak
için bir Jev çağrısı yapar; bkz. Kararlar bölümü). `ask`, `chat` ve `summarize` ise
Claude'a ihtiyaç duyar; RepoFinder Claude'a iki yolla ulaşabilir. Birini seçin.

**A. Anthropic API anahtarı** (kullandıkça öde). <https://console.anthropic.com>
adresinde bir anahtar oluşturun ve RepoFinder'ın görebileceği hale getirin:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."          # PowerShell
set ANTHROPIC_API_KEY=sk-ant-...                # cmd.exe
export ANTHROPIC_API_KEY=sk-ant-...             # macOS / Linux
```

ya da `ANTHROPIC_API_KEY=sk-ant-...` satırını RepoFinder'ı çalıştırdığınız
dizindeki bir `.env` dosyasına yazın.

**B. Claude Code üzerinden Claude aboneliğiniz.** Claude Code'u kurun
(<https://claude.com/claude-code>), `claude auth login` ile bir kez oturum açın
(ya da `claude`'u başlatıp `/login` yazın) ve `claude` komutunun `PATH`
üzerinde olduğundan emin olun. Başka bir şey gerekmez: RepoFinder
her model çağrısı için `claude -p` çalıştırır; bu da API faturalandırması
yerine aboneliğinizin limitlerini kullanır.

**Otomatik algılama.** Bir API anahtarı (ya da başka Anthropic SDK kimlik
bilgileri) bulunursa A seçeneği kullanılır; aksi halde RepoFinder `claude`
komutunu arar. Açıkça seçmek için `REPOFINDER_BACKEND=anthropic` ya da
`REPOFINDER_BACKEND=claude-code` ayarlayın. `repofinder status <repo>`
hangisinin kullanılacağını gösterir. İkisi de yoksa LLM komutları kurulum
talimatlarını yazdırır ve 2 koduyla çıkar.

Varsayılan olarak dosya başına özetler için `claude-sonnet-5-5`, yanıtlar ve
genel bakış için `claude-opus-5-5` kullanılır; bunları `REPOFINDER_MODEL_FAST`
ve `REPOFINDER_MODEL_SMART` ile değiştirebilirsiniz.

### Kararlar (isteğe bağlı, Jev)

Sohbet modu, Claude yanıtı yazmadan önce her turda birkaç küçük karar verir:
mesajın ne tür olduğu ("nerede" sorusu, açıklama, takip sorusu, gündelik
sohbet), hangi arama sonucunun onu yanıtladığı ve daha önce gösterilen
kesitlerin bir takip sorusunu karşılayıp karşılamadığı. Tek seferlik `ask`
bunlardan yalnızca birini verir: soruyu hangi sonucun yanıtladığı. Bu kararları
yerleşik İngilizce ve Türkçe kurallar çevrimdışı ve ücretsiz olarak verebilir;
kurallar arama sonuçlarının sırasını asla değiştirmez. Metin yerine
olasılıklarla birlikte tipli seçenekler döndüren TypeSafe'in Jev modeli de
kurulduğunda bu kararları verir; kurulduktan sonra varsayılan olarak Jev
kullanılır. Jev SDK'sı (`typesafe-sdk`) RepoFinder ile birlikte gelir; yalnızca
bir API anahtarı gerekir:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."               # OpenRouter üzerinden ya da
$env:TYPESAFE_API_KEY = "..."                       # TypeSafe'in kendi API'si üzerinden
$env:REPOFINDER_DECISIONS = "auto"                  # isteğe bağlı; aşağıya bakın
```

`REPOFINDER_DECISIONS` ayarlanmamışsa (ya da `auto` ise), `OPENROUTER_API_KEY`
(ya da `TYPESAFE_API_KEY`) ayarlıysa Jev, değilse
sessizce yerleşik kurallar kullanılır. `rules` (ya da `off`) anahtar olsa bile
kuralları zorlar; `jev` ise Jev'i zorunlu kılar. Bu `ask`, `chat` ve `search` için
geçerlidir.

Anahtarlar `.env` dosyasına da yazılabilir. İkisi birden ayarlıysa OpenRouter
anahtarı kullanılır (model `~typesafe/jev-latest`). Jev yalnızca girdiyi
ücretlendirir (milyon token başına yaklaşık $0.042; çıktı ücretsizdir) ve her
istek modelin 32k token'lık bağlamının altında tutulur. Anahtar olmadan
`REPOFINDER_DECISIONS=jev` zorlamak bir yapılandırma
hatasıdır ve bu kurulum adımlarını yazdırır.

## Hızlı başlangıç (Flask)

```powershell
repofinder index https://github.com/pallets/flask
repofinder map   https://github.com/pallets/flask --out flask-map.md
repofinder ask   https://github.com/pallets/flask "Where is the route decorator implemented?"
```

`index` sığ bir klon (shallow clone) alır ve indeksi oluşturur (Flask için
klonlama süresi hariç bir saniyenin altında).
Diğer tüm komutlar da ilk kullanımda indeksleme yapar; bu yüzden tek başına
`ask` da çalışır. URL kabul eden her yerde yerel bir dizin de verilebilir; bu
dizin kopyalanmadan yerinde indekslenir.

`ask` yanıtı, bir **Sources** listesini ve harcanan süre ile token'ları
yazdırır:

```
The route decorator is implemented in Scaffold.route in
src/flask/sansio/scaffold.py:343-373.
...
Sources
  src/flask/sansio/scaffold.py:343-373
  src/flask/sansio/scaffold.py:461-463

7.6 s · 1 call · 4,936 input + 656 output tokens · 567 cache read
```

Sorular herhangi bir dilde sorulabilir. Yanıt, sorunun dilinde gelir.

İsterseniz `repofinder summarize <repo>` çalıştırarak Claude'a her dosyanın
kısa bir özetini ve deponun genel bakışını yazdırabilirsiniz. Özetler haritayı
zenginleştirir ve aramaya yardımcı olur; genel bakış ise `ask`'in yönünü
bulmasına yardım eder. Dosya başına bir model çağrısı yapar (Flask için
yaklaşık 80), bu yüzden önce `--limit 10` ile deneyin.

## Komutlar

| Komut | Ne yapar |
| --- | --- |
| `repofinder index <source> [--refresh]` | Klonlar (URL ise) ve indeksler. `--refresh` mevcut bir indeksi yeniden oluşturur ve hâlâ var olan dosyaların özetlerini korur. |
| `repofinder map <source> [--out FILE] [--max-modules N]` | Markdown biçiminde mimari haritası: giriş noktaları, fan-in'e göre çekirdek modüller, paket düzeni, sıcak noktalar ve nereden başlanacağı. |
| `repofinder ask <source> "question" [-k N] [--show-context] [--decisions jev\|rules] [--candidates N] [--json]` | Alıntılarla yanıt verir. `-k` kaç arama sonucunun bağlama gireceğini belirler (varsayılan 8); `--show-context` Claude'a gönderilen kesitleri listeler; `--decisions` bu çalıştırma için karar katmanını seçer (varsayılan: kuruluysa Jev, değilse kurallar); Jev ile, en üstteki `-k` sonuç bağlama girmeden önce tek bir çağrı ilk `--candidates` sonucu yeniden sıralar (varsayılan 16, en fazla 16); `--json` şunları yazdırır: `answer`, `citations`, `hits`, `chunks`, `model`, `backend`, `usage`, `elapsed_s`, `decisions` (`name`, `calls`, `cost_usd`, `reranked`, `off_target`, `error`), `candidates` ve `probabilities` (`path:start-end`'den p'ye). |
| `repofinder chat <source> [-k N] [--decisions jev\|rules] [--json]` | Kod hakkında bir sohbet; stdin'den her satırda bir mesaj okunur (bkz. [Sohbet modu](#sohbet-modu)). `-k` arama başına sonuç sayısını belirler (varsayılan 8); `--decisions` bu çalıştırma için karar katmanını seçer. `/context` son aramanın kesitlerini listeler, `/route` son mesajın nasıl yönlendirildiğini gösterir, `/help` komutları listeler, `/quit` ya da girdinin sonu sohbetten çıkarır. `--json` her tur için satır başına bir JSON nesnesi yazdırır (`route`, `answer`, `citations`, `reused_context`, `widened`, `usage`, `decisions`, ...). `you>` istemi yalnızca terminalde yazarken gösterilir; böylece pipe ile aktarılan çıktı temiz kalır. |
| `repofinder summarize <source> [--limit N] [--all]` | Önce dosya başına özetler, ardından deponun genel bakışı. `--all` verilmedikçe yalnızca özeti olmayan dosyalar işlenir. Art arda 3 hatadan sonra durur (genellikle bir kullanım limiti); tamamlanan iş korunur. |
| `repofinder status <source>` | Dosya, sembol ve import sayıları, özetler, genel bakışın olup olmadığı, indekslemeden bu yana değişen dosyalar, LLM backend'i, karar katmanı, veri dizini. `Decisions` satırı `jev (OPENROUTER_API_KEY set)` ya da `rules (no OPENROUTER_API_KEY / TYPESAFE_API_KEY; set one in .env to enable Jev)` olarak görünür; `REPOFINDER_DECISIONS=rules` ayarı ve SDK import hatası için ayrı varyantlar vardır. |
| `repofinder search <source> "query" [-k N] [--decisions jev\|rules] [--candidates N]` | Arama sonuçları, puanları ve neden eşleştikleriyle birlikte; karar katmanı Jev ise Jev'in sırasıyla (yanıt yok; bir yanıt bir şeyi kaçırdığında işe yarar). Jev ile `max(-k, --candidates)` sonuç aranır (`--candidates` varsayılanı 16), ilk `--candidates` tanesi tek çağrıda yeniden sıralanır, ilk `-k` Jev'in sırasıyla yazdırılır, Jev'e gösterilen her kartın ardından `p=0.80` gibi bir olasılık gelir ve son satır `decisions: jev (1 call, $0.000165) · reranked` biçimindedir. Kurallarla (anahtar yok) çıktı düz arama sonuçlarıdır ve böyle bir satır yazdırılmaz. |

Her komut `--data-dir PATH` kabul eder. İndeksler varsayılan olarak
`~/.repofinder` altında (ya da `$REPOFINDER_HOME` içinde) tutulur:
`repos/<name>/index.db`, klonlanan depolar için ayrıca
`repos/<name>/checkout/<name>/`. Windows PowerShell 5'te haritayı `>` yerine
`--out` ile yazın; bu sürümdeki yönlendirme çıktının kodlamasını
değiştirir.

### Sohbet modu

`repofinder chat <repo>` tek bir depo hakkında sohbet yürütür: karar katmanı
karar verir, Claude konuşur. Her mesaj önce yönlendirilir. Bir "nerede" sorusu
(*locate*) ya da "nasıl çalışıyor" sorusu (*explain*) bir arama başlatır, karar
katmanının en ilgili bulduğu kesitleri başa koyar ve hiçbirini ilgili bulmazsa
üç kat fazla sonuçla bir kez daha arar. Bir takip sorusu (*follow-up*), karar
katmanı önceki kesitlerin onu karşıladığını söylerse bu kesitleri yeniden
kullanır; aksi halde önceki soruyu yeni mesajla birlikte yeniden arar. Gündelik
sohbet (*chat*) ve deponun başka bir kütüphaneye bıraktığı bir konuyla ilgili
sorular (*outside*) arama yapılmadan kısa bir yanıt alır. Claude
(`REPOFINDER_MODEL_SMART`) her yanıtı `ask` ile aynı `path:start-end` alıntı
kurallarıyla yazar ve sohbetin son altı mesajını görür. Her yanıttan sonra soluk
bir satır; yönlendirmeyi, kesitlerin yeniden kullanılıp kullanılmadığını ya da
aramanın genişletilip genişletilmediğini, süreyi, token'ları ve karar
çağrılarını gösterir.

Kararları, [Kararlar (isteğe bağlı, Jev)](#kararlar-isteğe-bağlı-jev) bölümünde
anlatıldığı gibi, kuruluysa Jev, değilse yerleşik kurallar verir; kurallarla
sohbet modu Claude'un yanıtı dışında hiçbir model çağrısı yapmaz.

### Jev ile ask

`ask`, Backend satırında `decisions: <name>` gösterir. Jev çalıştığında kullanım
satırına `decisions: jev (1 call, $0.000xxx) · reranked` (ya da `· off-target`)
eklenir. Arama `max(k, candidates)` sonuç getirir; ilk `candidates` sonuç
(varsayılan 16, asla 16'dan fazla değil) kartlara dönüştürülür
(`path:start-end`, bir etiket ve ilk 40 satır) ve bir NONE seçeneğiyle birlikte
tek çağrıda Jev'e gönderilir. Seçilen sıra uygulanır ve en üstteki `k` sonuç
bağlam olur. Jev NONE seçerse arama sırası korunur ve yanıt konu dışı
(off-target) olarak işaretlenir. Çağrı başarısız olursa arama sırası korunur ve
hata raporlanır (`--json` içinde `decisions.error`); anahtar olmadan zorlanmış
bir `jev` gibi bir yapılandırma hatası ise bunun yerine işlemi durdurur.
`pipeline.retrieve` deterministik ve değişmeden kalır; `pipeline.rerank_retrieve`
katmanı ekler ve `pipeline.ask` onu kullanır. Altın setlerde 16 adaylı Jev, soru
başına 0.02 sentin altında bir maliyetle Flask'ta hit@1'i 0.67'den 0.89'a,
requests'te 0.59'dan 0.88'e çıkarır
([ayrıntılar](eval/README.md#jev-in-the-pipeline-2026-10-12)).

## Nasıl çalışır

Tree-sitter her Python dosyasını sınıflara, fonksiyonlara, metotlara ve
import'lara ayrıştırır; bunlar bir import grafiğiyle (fan-in, fan-out, giriş
noktaları) birlikte bir SQLite veritabanında saklanır. Bir soru, en iyi
sembolleri ve dosyaları döndüren sözcüksel bir aramayla (sembol adları,
imzalar, docstring'ler, yollar ve dosya özetleri üzerinde BM25) yanıtlanır. Bir
sembolün ad takviyesi, adının sorunun ne kadarını kapsadığına göre ölçeklenir;
böylece tesadüfen bir ad olan yaygın bir sözcük gerçek yanıtın önüne geçmez.
Soru bir metodun sınıfını adıyla andığında metot da puan kazanır. Sözcükler
hafif bir İngilizce kök bulma (stem) ve yaygın tanımlayıcı kısaltmaları
üzerinden eşleştirilir ("python file" sorgusu `from_pyfile` sembolünü bulur);
Türkçe sorular ise Türkçe dolgu sözcükleri atılıp geri kalanı küçük bir
sözlükle çevrilerek desteklenir. Seçilen sembollerin tam kaynak satırları
kesitlere dönüşür (fonksiyon ve sınıfların tamamı, uzun olanlar yaklaşık 100
satırlık pencerelerle, metotlar için sınıf başlığı), toplamda yaklaşık 36,000
karakter. Claude yalnızca bu kesitlerden yanıt verir ve `path:start-end`
biçiminde alıntı yapar; Sources listesi yalnızca kesitlerin içine düşen
alıntıları tutar.

## Sınırlamalar

- Şimdilik yalnızca Python (diğer diller için bir ayrıştırıcı kaydı gerekir).
- Getirme işlemi anlamsal değil sözcükseldir: sorunun kodla ortak sözcükleri
  olmalıdır (tanımlayıcılar, docstring'ler, dosya adları). Yanıt yanlış
  görünüyorsa `repofinder search` neyin getirildiğini gösterir; beklediğiniz
  adlarla yeniden ifade etmek ya da İngilizce sormak genellikle işe yarar.
- Türkçe desteği, yaygın kod sözcüklerinden oluşan küçük bir yerleşik sözlükle
  sınırlıdır; Türkçe harfler olmadan yazılan Türkçe (ASCII olarak yazılmış
  "nasil", "kaydediliyor") yalnızca kısmen çalışır.
- Yanıt yalnızca getirilen kesitleri bilir. Kesitler yanıtı içermiyorsa bunu
  söyler ve sonra nereye bakılacağını önerir.
- Aynı ada sahip iki depo aynı veri dizinini paylaşır; birini açmak diğerinin
  indeksinin yerine geçer.

## Geliştirme

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
$env:REPOFINDER_SKIP_LIVE = "1"; .venv\Scripts\python -m pytest -q
```

`REPOFINDER_SKIP_LIVE=1`, gerçek `claude` CLI'ını çağıran birkaç testi atlar.
Geri kalan her şey sahte backend'lerle çevrimdışı çalışır.
`tests/fixtures/sample_repo/` bir test paketi değil, test verisidir
(modüllerinden birinde kasıtlı bir sözdizimi hatası vardır).

Kod `repofinder/` altındadır: `ingest` ve `parser` (klonlama, dosya gezme,
tree-sitter), `store` (SQLite), `search` (BM25), `graph` ve `mapper` (import
grafiği, harita), `llm`, `prompts` ve `backends` (Claude), `chunker`
(sonuçlardan kesitlere), `pipeline` (CLI'ın ve değerlendirmenin çağırdığı API;
`retrieve` deterministik aramadır, `rerank_retrieve` karar katmanını ekler),
`chat` ve `decisions` (sohbet modu ve onun yönlendirme seçimleri) ve `cli`.

Katkıda bulunanlar için notlar:

- `Symbol.id` (`path::qualname`) benzersiz değildir: bir property'nin getter ve
  setter'ı, `@overload` stub'ları ve koşullu yeniden tanımlar aynı id'yi
  paylaşır. Anahtar olarak `(id, start_line)` kullanın.
- `FileRecord.parse_ok=False`, tree-sitter'ın bir sözdizimi hatası bildirdiği
  anlamına gelir; dosyanın sembolleri eksik olabilir.
- Tree-sitter noktalarını `.row` ya da `.column` ile değil, indeksle okuyun
  (`node.start_point[0]`): py-tree-sitter 0.26.0'da bunlar 256'nın üzerindeki
  değerlerde belleği bozar (ayrıntılar `parser.py` içinde).
- Klonlanan depolar güvenilmeyen verilerdir. RepoFinder onları yalnızca okur,
  kodlarını asla çalıştırmaz.

## Lisans

MIT © Selman Boray Asiltürk, bkz. [LICENSE](LICENSE).
