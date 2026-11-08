# RepoFinder
*[English](README.md)*

RepoFinder, büyük bir açık kaynak projeye katkıda bulunmadan önce o projede
yolunuzu bulmanıza yardım eder. Ona bir GitHub deposu verdiğinizde bir mimari
haritası çizer ve "X özelliği nerede uygulanmış?" sorusunu koda işaret eden
`path:start-end` alıntılarıyla yanıtlar. Python, TypeScript, JavaScript, Go, Rust ve Java projelerini okur (Dart isteğe bağlı).

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

Varsayılan olarak her iş (dosya başına özetler, yanıtlar ve genel bakış) için
`claude-sonnet-5-5` kullanılır; bunları `REPOFINDER_MODEL_FAST`
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
| `repofinder index <source> [--refresh]` | Klonlar (URL ise) ve Python, TypeScript, JavaScript, Go, Rust ve Java dosyalarını (`dart` ekstrasıyla Dart da) indeksler. `--refresh` mevcut bir indeksi yeniden oluşturur ve hâlâ var olan dosyaların özetlerini korur. |
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

Tree-sitter her Python, TypeScript, JavaScript, Go, Rust ve Java dosyasını (kuruluysa Dart da) sınıflara, fonksiyonlara,
metotlara ve import'lara ayrıştırır; bunlar bir import grafiğiyle (fan-in, fan-out, giriş
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
satırlık pencerelerle, metotlar için sınıf başlığı; Java, Go ve Rust metotlarında sahibinin başlığı, bkz. [Diller](#diller)), toplamda yaklaşık 36,000
karakter. Claude yalnızca bu kesitlerden yanıt verir ve `path:start-end`
biçiminde alıntı yapar; Sources listesi yalnızca kesitlerin içine düşen
alıntıları tutar.

### Diller

Python (`.py`), TypeScript (`.ts`, `.mts`, `.cts`, `.tsx`), JavaScript, Go,
Rust ve Java indekslenir; Dart isteğe bağlıdır (aşağıda).
TypeScript sembolleri fonksiyonlar, sınıflar, metotlar, interface'ler,
type'lar, enum'lar, namespace'ler ve arrow-function const'larıdır; JSDoc
yorumu docstring olarak alınır. Göreli import'lar çözülür (`./a.js`,
`a.ts` veya `index.ts` dosyasını bulur), `index.ts` adlı dosya dizinin adını
taşır ve `*.test.ts`, `*.spec.ts` ile `__tests__/` test sayılır.
`node_modules`, `dist`, `build`, `third_party`, `licenses`, minify edilmiş
dosyalar ve 2 MB'tan büyük dosyalar atlanır. morluto/rea üzerinde (3.119
dosya, 747 bin satır) indeksleme 5,4 sn sürdü ve 12,5 MB'lık indekste 15.267
sembol üretti; 59 dosya ayrıştırılamadı (tree-sitter 0.23
`importOriginal<typeof import("x")>()` ifadesini okuyamıyor); arama anında
yanıt verir. (Bu rakamlar JavaScript indekslenmeden önce alındı; REA'nın 420
JavaScript dosyası artık indeksleniyor.)

JavaScript (`.js`, `.jsx`, `.mjs`, `.cjs`; gramer `tree-sitter-javascript`,
JSX dahil) TypeScript çıkarıcısını paylaşır; yani aynı sembolleri verir, ek
olarak fonksiyon tutan sınıf alanlarını ve yalnızca JavaScript kodunda görülen
atamaları da alır: `X.prototype.m = function` `X.m` metodudur, `exports.f = ...`
ve `module.exports.f = ...` `f` fonksiyonlarıdır, `obj.f = function` `obj.f`
qualname'li bir fonksiyondur, `module.exports = function name() {}` ise `name`
olur (adsız olana sembol verilmez). Göreli import belirteçleri TypeScript
kurallarıyla çözülür: her dosya adayı herhangi bir dizin indeksinden önce
gelir, TypeScript JavaScript'ten önce denenir ve indeksliyse yazıldığı haliyle
belirtilen dosya kazanır. `require()` ve `import()` çağrıları da import
kenarı üretir. Aynı gövde adlı bir TypeScript kaynağının yanındaki `.js`,
`.jsx`, `.mjs` veya `.cjs` dosyası (`a.ts` yanında `a.js`) derlenmiş çıktı
sayılır ve atlanır; bildirim dosyası (`.d.ts`) bu kaynak yerine geçmez.

Go (`.go`): fonksiyonlar, metotlar (`Recv.Name`), struct'lar (`class` olarak),
interface'ler (`interface` olarak, metotlarıyla) ve diğer tipler (`type`
olarak) semboldür; doc yorumu docstring olur. Bir import, depo içindeki en yakın
`go.mod` üzerinden, adını verdiği paketin test olmayan bir dosyasına çözülür;
standart kütüphane ve başka modüller çözümsüz kalır. Üretilmiş dosyalar atlanır
(`*.pb.go`, `*.pb.gw.go`, `zz_generated*` ve başlığında
`// Code generated ... DO NOT EDIT.` olan her dosya). `*_test.go`, `testdata/`
ve test dizinleri test sayılır; `main.go` ile `cmd/<name>/main.go` giriş
noktalarıdır. Derleme kısıtları (`//go:build`) yok sayılır; bir paketin her
varyantı sembollerini verir.

Rust (`.rs`): `fn`, `impl` bloklarındaki metotlar (`Type.method`), trait'ler
(`interface` olarak), struct ve union'lar (`class` olarak), enum'lar, tip
takma adları ve satır içi `mod` blokları (`namespace` olarak) semboldür;
aralık öğenin ilk attribute'unda başlar, `///` yorumları docstring'dir. Her
`use` yaprağı bir import kenarıdır, `mod` bildirimleri de öyle. `crate::`,
`self::` ve `super::` dosya yollarıyla çözülür; aynı workspace'teki diğer
crate'ler de `Cargo.toml` içindeki `[package]` adıyla bulunur. İzlenmeyenler:
`#[cfg]`, `#[path]`, makrolar ve yeniden dışa aktarma zincirleri
(`use crate::Foo`, `Foo`'yu yeniden dışa aktaran dosyaya çözülür). `Cargo.toml`
yanındaki `target/` dizini atlanır. Giriş noktaları `Cargo.toml`'daki
`[[bin]]` girdilerinden ve varsayılan `src/main.rs`'ten gelir; `lib.rs` her
çalıştırılabilir dosyanın altında sıralanır. Aramada sorudaki `a::b::C`
ifadesi `a.b.C` gibi okunur ve `mod.rs` dizininin adını taşır.

Java (`.java`): sınıflar ve record'lar (`class` olarak), interface'ler ve
`@interface` (`interface` olarak), enum'lar, iç içe tipler (`Outer.Inner`),
metotlar ve kurucular (`Type.Type`) semboldür; aşırı yüklemeler tek id
paylaşır, Javadoc docstring olur. Tekil, joker ve static import'lar yol
sonekiyle çözülür (`a.b.C`, `a/b/C.java` ile biten bir dosyadır); test olmayan
dosya, sonra en kısa yol tercih edilir. Modül import'ları (`import module ...`)
kenar üretmez. `pom.xml` yanındaki `target/` dizini atlanır. `*Test.java`,
`*Tests.java`, `*IT.java` ve `src/test` test sayılır.

Java, Go ve Rust metotlarında kesit sahibinin başlığını taşır: Java'da sınıf,
record, interface veya enum başlığı, Go'da aynı dosyada bildirilmişse alıcı
(receiver) tipi, Rust'ta `impl` satırı.

Gramer sürümleri `pyproject.toml`'da sabitlenmiştir (`tree-sitter>=0.25`;
JavaScript ve Go 0.25.x, Rust 0.24.x, Java 0.23.x), çünkü çıkarıcılar
gramerlerin düğüm adlarına bağlıdır.

Ölçülen kalite (katı, k=8, hit@1 / hit@8): bu diller için yazılan ilk kör
altın setler, her biri 20 soru (12 İngilizce, 8 Türkçe). Arama bu setlere göre
ayarlanmadı:

| set (dil, commit) | arama | arama + Jev (16 aday) |
| --- | --- | --- |
| axios (JavaScript, `a81d62a`) | 0.59 / 0.77 | 0.71 / 0.77 |
| gin (Go, `0f09c3a`) | 0.35 / 0.82 | 0.82 / 0.94 |
| gson (Java, `65df6e5`) | 0.41 / 0.71 | 0.65 / 0.77 |
| ripgrep (Rust, `91f5d22`) | 0.12 / 0.53 | 0.59 / 0.59 |
| flame (Dart, `88c0438`) | 0.23 / 0.47 | 0.59 / 0.59 |

Bunlar ilk taban çizgileridir; tam sayılar ve ıskaların teşhisi için
[eval/README.md](eval/README.md) dosyasına bakın.

Dart (`.dart`) isteğe bağlıdır: `pip install -e ".[dart]"` ile kurulur. İlk
`.dart` dosyası işlendiğinde gramer indirilir (yaklaşık 22 MB, tek sefer,
sonra önbellekten), yani bu adım bir kez ağ gerektirir. Ekstra kurulu değilse
`.dart` dosyaları atlanır; Python ve TypeScript etkilenmez. Dart 3.13
sözdizimi okunur (primary constructor dahil). Üretilmiş dosyalar
(`*.g.dart`, `*.freezed.dart`, `*.mocks.dart`, `*.gr.dart`, `*.pb.dart`)
atlanır; liste yalnızca bunlardır, diğer üretilmiş Dart dosyaları indekslenir.
Mixin, extension ve extension type sınıf olarak; getter, setter,
operator ve kurucular metot olarak indekslenir. Üst düzey değişkenler sembol
olarak çıkarılmaz. Dart'ta arama kalitesi Flame üzerinde bir kez ölçüldü
(yalnız aramayla hit@1 0.23, hit@8 0.47; yukarıdaki tabloya bakın).

`dart` ekstrasının güven ve çevrimdışı kullanım açısından anlamı: gramer,
`tree-sitter-language-pack`'in bir GitHub release'inden indirdiği ve yerel kod
olarak yüklediği bir kütüphanedir. RepoFinder dosyayı yüklemeden önce, paket
sürümü 1.23.1 için kodda sabit SHA-256 özetleriyle denetler. Özetler 6
platformu kapsar: Windows x86_64 ve aarch64, Linux x86_64 ve aarch64, macOS
arm64 ve x86_64. Yalnız Windows x86_64 çalıştırılarak denendi; diğerlerinde
özetler release'e karşı doğrulandı ama gramer çalıştırılmadı. Paket sürümü
1.23.1 değilse Dart grameri kapalı kalır. Önbellekteki dosyanın özeti
uyuşmazsa, sürüm beklenen değilse ya da platformun özeti sabitlenmemişse gramer
yüklenmez ve bir uyarı yazılır. Bu bir özet denetimidir, imza doğrulaması
değildir; ikiliye, özetin sabitlendiği andaki haliyle güvenir. Denetim,
değiştirilmiş bir release dosyasına ve önceden bozulmuş ya da değiştirilmiş bir
önbellek dosyasına karşı korur; önbelleğinize aynı anda yazabilen birine karşı
korumaz, çünkü ilk indirmede ve Linux/macOS'ta denetimle yükleme arasında çok
kısa bir an kalır. Önbellekteki
dosya denetimi geçmezse yerinde bırakılır ve uyarı onu adıyla anar; release'in
dosyası olması gerekiyorsa yeniden inmesi için silinir. Hata süreç boyunca
hatırlandığı için ardından `repofinder index --refresh` yeniden çalıştırılır.
Gramer indirilemezse ya da yüklenmezse `.dart` dosyaları indekse sembolsüz
kaydedilir; neden giderilince (ağ gelince ya da bozuk dosya silinince)
`repofinder index --refresh` bunu düzeltir.

## Sınırlamalar

- Şimdilik yalnızca Python, TypeScript, JavaScript, Go, Rust ve Java (isteğe
  bağlı Dart); diğer diller için bir ayrıştırıcı kaydı gerekir.
- Arama kalitesi dil başına ölçülür (bkz.
  [eval/README.md](eval/README.md#javascript-go-rust-and-java-first-baselines-2026-11-08))
  ve eşit değildir. Rust (ripgrep) ve Dart (flame) için yazılan ilk kör
  setlerde sorunun yaklaşık %40'ında yanıt ilk 16 adayın dışında kalıyor ve
  Jev aramanın getirmediği şeyi kurtaramıyor. Bilinen, henüz giderilmemiş
  nedenler: Rust satır içi testleri (`#[cfg(test)] mod tests`, `#[test]`) test
  sayılmıyor; örnek, dokümantasyon, benchmark ve betik dizinlerinin arama
  puanı düşürülmüyor; Türkçe sözcük dağarcığında boşluklar var. REA'nın
  TypeScript sonuçları, JavaScript dosyaları (betikler, deneyler, fixture'lar)
  indekslenmeye başlayınca biraz geriledi.
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
  setter'ı, `@overload` stub'ları, koşullu yeniden tanımlar, Java aşırı
  yüklemeleri, Go `init` fonksiyonları ve Rust'ta aynı metot adlı trait
  impl'leri aynı id'yi paylaşır. Anahtar olarak `(id, start_line)` kullanın.
- `FileRecord.parse_ok=False`, tree-sitter'ın bir sözdizimi hatası bildirdiği
  anlamına gelir; dosyanın sembolleri eksik olabilir.
- Tree-sitter noktalarını `.row` ya da `.column` ile değil, indeksle okuyun
  (`node.start_point[0]`): py-tree-sitter 0.26.0'da bunlar 256'nın üzerindeki
  değerlerde belleği bozar (ayrıntılar `parser.py` içinde).
- Klonlanan depolar güvenilmeyen verilerdir. RepoFinder onları yalnızca okur,
  kodlarını asla çalıştırmaz.

## Lisans

MIT © Selman Boray Asiltürk, bkz. [LICENSE](LICENSE).
