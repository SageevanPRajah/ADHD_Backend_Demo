import streamlit as st
import librosa
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from io import BytesIO
import tempfile
import os
from src.preprocessing.audio_preprocessor import preprocess_audio
from src.feature_extraction.feature_extractor import extract_features, transcribe_audio
from src.models.classifier import classify_adhd

st.set_page_config(page_title="ADHD Speech Detection", page_icon="🎤", layout="wide")

st.title("🎤 ADHD Speech Detection for Sinhala-speaking Children")
st.markdown("---")

# Sidebar for user input
with st.sidebar:
    st.header("Settings")
    child_age = st.number_input("Child's Age", min_value=6, max_value=12, value=8, step=1)
    st.markdown("---")
    st.info("This tool analyzes speech patterns to screen for ADHD indicators.")

# Generate age-appropriate paragraph (2 minutes reading time - approximately 200-250 words)
def generate_paragraph(age):
    paragraphs = {
        6: """මම ගෙදර බල්ලෙක් හදනවා. එයාගේ නම බඩී. බඩී හරිම ආදරෙයි. මම හැම දාම එයා එක්ක සෙල්ලම් කරනවා. බඩී කුඩා විශාල බල්ලෙක්. එයාට දුවන එක හරිම ආසයි. අපි උයනේ දුවනවා. බඩී බෝලයක් එක්ක සෙල්ලම් කරනවා. එයා බෝලය අල්ලගෙන මා වෙත ගෙනෙනවා. මම එය විසි කරනවා. බඩී ආයෙත් ගෙනෙනවා. අපි මේක නැවත නැවත කරනවා.
        
සවස වෙලාවට මම බඩීට කෑම දෙනවා. එයා හරිම සතුටු වෙනවා. එයාගේ වලිගය හෙල්ලනවා. බඩී මගේ හොඳම යාළුවා. මම ඔහුට බොහෝ ආදරෙයි. ඔහු මට පෙන්වනවා කෙසේද පුරුදු වන්නේ කියලා. මම බඩීට "අත දෙන්න" කියලා උගන්වනවා. ඔහු මගේ අත උඩ ඔහුගේ පාදය තියනවා. ඔහු හරිම දක්ෂයි.
        
සති අන්තයට අපි හැමෝම උයන් කඩයට යනවා. බඩීත් එනවා. එතන බඩී අනිත් බල්ලන් එක්ක ගැටෙනවා. ඔහු නව යාළුවන් හොයනවා. බඩීට වතුර ජෙට් එකෙන් සෙල්ලම් කරන්න ආසයි. ඔහු තෙත් වෙනවා. ඔහු මා උඩට පනිනවා. මම ඔහුව වළක්වන්න බැහැ. අපි සියලු දෙනාම සිනාසෙනවා.
        
රාත්‍රියේ බඩී මගේ ඇඳ ලඟ නිදාගන්නවා. ඔහුට ආරක්ෂිත හැඟීමක් දැනෙනවා. මම ආරක්ෂිත හැඟීමක් දැනෙනවා. බඩී මගේ පවුලේ කොටසක්. අපි ඔහුව හරිම ආදරෙයි.""",
        
        7: """මගේ ප්‍රියතම ක්‍රීඩාව පාපන්දු. මම හැම සතියේම මගේ යාළුවන් එක්ක පාපන්දු ක්‍රීඩා කරනවා. මම ගෝල් දාන එක ලොකු ආසයි. අපේ කණ්ඩායම ඉතා හොඳයි. අපි එකට පුහුණු වෙනවා. අපේ පුහුණුකරු හරිම දක්ෂයි. ඔහු අපිට නව කෞශල උගන්වනවා.
        
පසුගිය සතියේ අපට විශාල තරඟයක් තිබුණා. අපි අපේ ප්‍රතිවාදීන්ට එරෙහිව ක්‍රීඩා කළා. තරඟය ඉතා උද්වේගකර විය. පළමු භාගයේ අපි එක ගෝලයක් ලබා ගත්තා. මම එය ගැන ඉතා සතුටු වුණා. නමුත් අනෙක් කණ්ඩායමත් මහන්සි වුණා. ඔවුන්ත් ගෝලයක් ලැබුවා. ලකුණු තත්ත්වය එක-එකයි.
        
දෙවන භාගයේ අපි තව මහන්සි වුණා. අපි වඩා හොඳින් දුවන්න පටන් ගත්තා. අපි බෝලය හොඳින් පාස් කළා. මගේ යාළුවා සමීර විශිෂ්ට පාස් එකක් මට දුන්නා. මම බෝලය අල්ලා ගත්තා. මම ගෝලය දෙසට දිව්වා. මම වෙඩි තැබුවා. බෝලය දැලට ගියා! අපි ජයග්‍රහණය කළා!
        
තරඟයෙන් පසු අපි සියල්ලෝම සතුටු වුණා. අපි අපේ ජයග්‍රහණය සමරනවා. අපේ පුහුණුකරු අපි ගැන ආඩම්බර විය. ඔහු කිව්වා අපි හොඳින් කළා කියලා. පාපන්දු මට කණ්ඩායම් වැඩ ඉගැන්වනවා. එය මට නව යාළුවන් හදන්න උදව් කරනවා. මම පාපන්දු ගැන ඉතා කැමතියි. මම හැමදාම ක්‍රීඩා කරන්න කැමතියි.""",
        
        8: """මම පාසලට යන්න ලොකු ආසයි. එතන මගේ යාළුවෝ ඉන්නවා. මම විද්‍යාව හා ගණිතය ඉගෙන ගන්න ලොකු කැමතියි. ගුරුවරු හරිම හොඳයි. අපේ පන්තිය විශාල සහ දීප්තිමත්. බිත්ති වල ලස්සන පින්තූර තියෙනවා. අපි පුස්තකාලයක් තියෙනවා. පුස්තකාලයේ බොහෝ පොත් තියෙනවා.
        
සෑම උදෑසනකම අපි ගණිතයෙන් පටන් ගන්නවා. මට ගණිතය ගැන ඉතා කැමතියි. අපේ ගුරුවරිය සරලව දේවල් පැහැදිලි කරනවා. ඇය අපිට ප්‍රශ්න විසඳන්න උදව් කරනවා. අපි එකට වැඩ කරනවා. අපි ගණනය කිරීම ඉගෙන ගන්නවා. අපි ගුණ කිරීම ඉගෙන ගන්නවා. මම ගණිත ප්‍රශ්න විසඳන්න ඉතා කැමතියි.
        
විද්‍යා පන්තියේ අපි අත්හදා බැලීම් කරනවා. අපි පරීක්ෂණ සිදු කරනවා. පසුගිය සතියේ අපි ශාක ගැන ඉගෙන ගත්තා. අපි බීජ සිටුවනවා. අපි ඒවා දිනපතා ජලය දෙනවා. බීජ වැඩෙනවා. එය ඉතා සිත්ගන්නාසුළුයි. අපි ශාකවලට ආලෝකය සහ ජලය අවශ්‍ය බව ඉගෙන ගත්තා.
        
දිවා ආහාර වේලාවේදී අපි එළිමහනේ සෙල්ලම් කරනවා. මම මගේ යාළුවන් සමඟ දුවනවා. අපි සැඟවීම සහ සොයන්න සෙල්ලම් කරනවා. සමහර වේලාවට අපි පොත් කියවනවා. පුස්තකාලය ශාන්ත සහ සිසිල්. මම කතා පොත් කියවන්න ඉතා කැමතියි.
        
පාසලෙන් පසුව අපට ගෙදර වැඩ තියෙනවා. මම මගේ ගෙදර වැඩ හොඳින් කරනවා. මගේ දෙමාපියෝ මට උදව් කරනවා. පාසල මට නව දේවල් ඉගෙන ගන්න උදව් කරනවා. එය මට වැඩෙන්න උදව් කරනවා. මම පාසලට යන්න ආදරෙයි.""",
        
        9: """සති අන්තයේ මම මගේ පවුලේ අය එක්ක සත්තු උද්‍යානයට ගියා. එතන මම අලි, වලසුන් හා වඳුරන් දැක්කා. ඒ හැම දෙයක්ම හරිම සිත්ගන්නා සුළු වුණා. අපි උදේ පාන්දර වගේ ගියා. දවස අලුයම්. අපි නගරයෙන් එකතිරියක් දුර ගියා. සත්ව උද්‍යානය විශාලයි. ඒ වනයක වගේ.
        
පළමුව අපි අලි දැක්කා. අලි ඉතා විශාලයි. ඔවුන් ඔවුන්ගේ දිගු හොරතුඩු තියෙනවා. එක් අලියෙක් ජලය ඉසිමින් සිටියේය. එය හරිම විනෝදජනක විය. කුඩා අලි පැටවු ඔවුන්ගේ මව අසල සෙල්ලම් කරමින් සිටියා. ඔවුන් ඉතා හුරුබුහුටියි. අපි බොහෝ වෙලාවක් ඔවුන් නරඹමින් ගත කළා.
        
ඊළඟට අපි වඳුරන් දැක්කා. වඳුරන් ගස්වල පැනගෙන යනවා. ඔවුන් ඉතා වේගවත්. සමහර වඳුරන් කෙසිලි කනවා. ඔවුන් අපේ දෙස බැලුවා. එක් වඳුරෙක් මාට දත් පෙන්වනවා. එය මාව සිනාසුණා. වඳුරන් හරිම දක්ෂයි.
        
අපි සිංහ කූඩුව පාර වුණා. සිංහයෝ විවේකයක් ගන්නවා. පිරිමි සිංහයාට විශාල මැන් එකක් තියෙනවා. ඔහු ඉතා ශක්තිමත් පෙනෙනවා. ගැහැණු සිංහිණියන් ඔවුන්ගේ පැටවුන් සමඟ තියෙනවා. පැටවුන් එකිනෙකා සමඟ සෙල්ලම් කරනවා. ඔවුන් මෘදු සහ හුරුබුහුටියි.
        
දිවා කෑමට අපි උද්‍යාන කැෆේ එකට ගියා. අපි සැන්ඩ්විච් සහ පලතුරු ආහාරයට ගත්තා. අපි මුළු දවසම ගැන කතා කළා. මගේ සහෝදරයා වඳුරන් ගැන ආදරෙයි. මම අලි ගැන ආදරෙයි. අපි සියලු දෙනාම විනෝදජනක කාලයක් ගත කළා.
        
සත්ව උද්‍යානයට යාම විශිෂ්ට දවසක් විය. අපි බොහෝ සතුන් ගැන ඉගෙන ගත්තා. අපි ඔවුන් ජීවත් වන ආකාරය දැක්කා. මම වෙනත් රටවල සතුන් ගැන ඉගෙන ගත්තා. සත්ව උද්‍යානය සතුන් ආරක්ෂා කරන්න උදව් කරනවා. අපි හැමෝම සත්තුන්ව රැකබලාගන්න ඕන කියලා මට හිතෙනවා.""",
        
        10: """මට පොත් කියවන එක ලොකු ආසයි. මම වික්‍රම සහ සාහසික කතා කියවනවා. කියවීම මගේ මනස වර්ධනය කරනවා හා මට නව දේවල් ඉගෙන ගන්න උදව් කරනවා. සෑම සතියකම මම පුස්තකාලයට යනවා. මම නව පොත් ලබා ගන්නවා. පුස්තකාලය මගේ ප්‍රියතම ස්ථානයයි. එහි මෙතරම් පොත් තියෙනවා.
        
මගේ ප්‍රියතම පොත වික්‍රම කතා පොතක්. එහි සාහසික කතා ගැන තියෙනවා. ප්‍රධාන චරිතය දක්ෂ ගමනාගමනකරුවෙක්. ඔහු බොහෝ ස්ථාන සංචාරය කරනවා. ඔහු මහාද්වීප හමුවෙනවා. ඔහු පුරාවිද්‍යා නටබුන් සොයනවා. සෑම පරිච්ඡේදයක්ම උද්වේගකරයි.
        
මම ප්‍රබන්ධ කතා කියවනවා. මම ප්‍රබන්ධ නොවන පොත් කියවනවා. මම ඉතිහාසය ගැන ඉගෙන ගන්න ඉතා කැමතියි. මම විද්‍යාව ගැන ඉගෙන ගන්න කැමතියි. පොත් මට ලෝකය ගැන උගන්වනවා. ඔවුන් මට විවිධ සංස්කෘතීන් ගැන උගන්වනවා. මම කියවන විට, මම මගේ පිළිබිඹුව භාවිතා කරනවා.
        
සමහර විට මම මගේ යාළුවන් සමඟ පොත් හුවමාරු කර ගන්නවා. අපි එකම පොත කියවනවා. ඉන්පසුව අපි ඒ ගැන කතා කරනවා. අපි කතාව ගැන කතා කරනවා. අපි චරිත ගැන කතා කරනවා. පොත් ගැන කතා කිරීම විනෝදජනකයි. එය මට වඩා හොඳින් තේරුම් ගන්න උදව් කරනවා.
        
රාත්‍රියේ මම නිදාගන්න කලින් කියවනවා. මම මගේ ඇඳේ අසුන් ගන්නවා. මම ලාම්පුව දල්වනවා. මම පොතක් විවෘත කරනවා. මම විනාඩි තිහක් කියවනවා. එය මට ලිහිල් වන්න උදව් කරනවා. එය මගේ මනස සන්සුන් කරනවා. කියවීම නිදාගැනීමට යන මාර්ගයයි.
        
මම විශාල වන විට, මට පොත් ලිවන්න අවශ්‍යයි. මට මගේම කතා නිර්මාණය කිරීමට අවශ්‍යයි. මට අනිත් අයට විනෝදාංශ කිරීමට අවශ්‍යයි. මට ඔවුන්ට නව දේවල් ඉගැන්වීමට අවශ්‍යයි. පොත් මගේ ලෝකය වෙනස් කළා. ඔවුන් මට සිහින දැකීමට උදව් කරනවා. කියවීම මගේ ආදරණීය විනෝදාංශයයි.""",
        
        11: """තාක්ෂණය අපේ ජීවිතවල හරිම වැදගත්. මම පරිගණක භාවිතා කරලා හොයාගන්නවා හා ඉගෙන ගන්නවා. අනාගතයේ මට සොෆ්ට්වෙයාර් ඉංජිනේරුවෙක් වෙන්න ඕනෑ. පරිගණක ගැන මට ඉතා කැමතියි. ඔවුන් පුදුම දේවල් කරන්න පුළුවන්. තාක්ෂණය ලෝකය වෙනස් කරනවා.
        
පාසලේ අපට පරිගණක විද්‍යා පන්තියක් තියෙනවා. අපි කේතකරණය ඉගෙන ගන්නවා. අපි වැඩසටහන් ලිවීම ඉගෙන ගන්නවා. මගේ ගුරුවරයා පරිගණක කේත රචනය කරන්න මට උගන්වනවා. අපි සරල ක්‍රීඩා සාදනවා. අපි චිත්‍ර නිර්මාණය කරන්න වැඩසටහන් සාදනවා. කේතනය ප්‍රහේලිකා විසඳීම වැනි ය.
        
මම නිවසේදී වැඩසටහන් අභ්‍යාස කරනවා. මම අන්තර්ජාලයේ නිබන්ධන බලනවා. මම නව කේත භාෂා ඉගෙන ගන්නවා. මම Python සහ JavaScript ඉගෙන ගන්නවා. මම කුඩා ව්‍යාපෘති සාදනවා. මම කැල්කියුලේටරයක් සෑදුවා. මම කාලගුණ යෙදුමක් සෑදුවා. සෑම ව්‍යාපෘතියක්ම මට නව දේවල් උගන්වනවා.
        
තාක්ෂණය අධ්‍යාපනය වෙනස් කරනවා. අපට ඕනෑම දෙයක් ඕනෑම වේලාවක ඉගෙන ගන්න පුළුවන්. අපට අන්තර්ජාලයේ පාඨමාලා තියෙනවා. අපට අධ්‍යාපනික වීඩියෝ තියෙනවා. අපට අන්තර්ක්‍රියාකාරී ක්‍රීඩා තියෙනවා. තාක්ෂණය ඉගෙනීම විනෝදජනක කරනවා. එය ඉගෙනීම පහසු කරනවා.
        
තාක්ෂණය සන්නිවේදනය වෙනස් කරනවා. අපට ලොව පුරා මිතුරන් සමඟ කතා කරන්න පුළුවන්. අපට වීඩියෝ ඇමතුම් කරන්න පුළුවන්. අපට ක්ෂණික පණිවිඩ යැවීමට පුළුවන්. තාක්ෂණය ජනතාව සම්බන්ධ කරනවා. එය ලෝකය කුඩා කරනවා.
        
අනාගතය තාක්ෂණයට අයත්. කෘතිම බුද්ධිය විවර වෙනවා. රොබෝවරු පොදු වෙමින් පවතී. අපට ස්වයං ධාවනය වන මෝටර් රථ ලැබෙයි. අපට ස්මාර්ට් නිවාස ලැබෙයි. මට මේ අනාගතයේ කොටසක් වෙන්න අවශ්‍යයි. මට හොඳ දේවල් සෑදීමට අවශ්‍යයි. මට තාක්ෂණය භාවිතා කර ගැටළු විසඳීමට අවශ්‍යයි. මට ලෝකය වඩා හොඳ තැනක් කිරීමට අවශ්‍යයි.""",
        
        12: """පරිසරය රැකගන්න අපි හැමෝම වගකිව යුතුයි. ප්ලාස්ටික් භාවිතය අඩු කරන්න, ගස් සිටුවන්න හා ජලය ඉතිරි කරන්න අපි තීරණය කරන්න ඕනෑ. අපේ ග්‍රහලෝකය අපේ නිවස. අපට එය අනාගත පරම්පරා සඳහා ආරක්ෂා කළ යුතුයි. දේශගුණික විපර්යාස සැබෑ ගැටළුවකි.
        
දේශගුණය වෙනස් වෙමින් පවතී. උෂ්ණත්වය ඉහළ යනවා. අයිස් තට්ටු දියවෙමින් පවතී. මුහුදු මට්ටම ඉහළ යනවා. අන්ත කාලගුණික සිදුවීම් සිදු වෙනවා. අපි වැඩි සිදුවීම් බලනවා. අපි වැඩි නියඟ බලනවා. මෙය අපේ ග්‍රහලෝකය සඳහා අනතුරුදායකයි.
        
අපේ ක්‍රියාවන් සැලකිය යුතු වෙනසක් කරන්න පුළුවන්. අපි විදුලිය ඉතිරි කරන්න පුළුවන්. අපි නැවත භාවිතා කළ හැකි බලශක්තිය භාවිතා කරන්න පුළුවන්. අපි ප්ලාස්ටික් අඩුවෙන් භාවිතා කරන්න පුළුවන්. අපි තව නැවත භාවිතා කරන්න සහ ප්‍රතිචක්‍රීකරණය කරන්න පුළුවන්. අපි තව ගස් පැළවීම කරන්න පුළුවන්. සෑම කුඩා ක්‍රියාවක්ම උදව් කරනවා.
        
අපේ පාසලේ අපි පරිසර ව්‍යාපෘතියක් ආරම්භ කළා. අපි පාසල් මිදුලේ ගස් සිටුවන්නවා. අපි සංයුක්තීකරණ පටල් ඇති කළා. අපි ප්ලාස්ටික් අඩු කිරීමට උත්සාහ කරන්නවා. අපි ප්ලාස්ටික් බෝතල් භාවිතා නොකරන්නවා. අපි නැවත භාවිතා කළ හැකි බෑග් ගෙන යන්නවා. අපේ ව්‍යාපෘතිය වෙනසක් සිදු කරමින් තියෙනවා.
        
ජලය ඉතිරි කිරීම ද වැදගත්. ජලය වටිනා සම්පතක්. බොහෝ ජනයාට පිරිසිදු ජලයට ප්‍රවේශය නැහැ. අපි ජලය නාස්ති නොකළ යුතුයි. අපි කෙටි නැවුම් ගන්න පුළුවන්. අපි කරාම සවි කරන්න පුළුවන්. අපි වැසි ජලය එකතු කරන්න පුළුවන්. ජලය ඉතිරි කිරීම පරිසරයට උදව් කරනවා.
        
වන ජීවීන් ආරක්ෂා කිරීම වැදගත්. බොහෝ සතුන් වඳවීමේ තර්ජනයට මුහුණ දෙනවා. ඔවුන්ගේ වාසස්ථාන විනාශ වෙමින් පවතී. අපි වන ජීවීන් ආරක්ෂා කළ යුතුයි. අපි වාසස්ථාන ආරක්ෂා කළ යුතුයි. අපි ජාතික උද්‍යාන සහ සංරක්ෂණ පිහිටුවීමට පුළුවන්. සෑම විශේෂයක්ම වැදගත්.
        
පරිසරය ආරක්ෂා කිරීම සැමගේම වගකීමයි. අපේ ක්‍රියාවන් අනාගතය තීරණය කරනවා. අපි අද ප්‍රතිකාර කළ යුතුයි. අපි කුඩා දේවල්වලින් ආරම්භ කරන්න පුළුවන්. අපට අපේ පවුල් හා යාළුවන් අධ්‍යාපනය ලබා දිය හැක. එකට, අපට වෙනසක් කරන්න පුළුවන්. අපි සෞඛ්‍ය සම්පන්න ග්‍රහලෝකයක් තනන්න පුළුවන්. එය අපගේ සහ අනාගත පරම්පරා සඳහා වේ."""
    }
    return paragraphs.get(age, paragraphs[8])

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📝 Reading Task")
    st.write(f"Please read the following paragraph aloud (Age {child_age}):")
    
    paragraph = generate_paragraph(child_age)
    st.info(paragraph)
    
    st.info(f"⏱️ **Estimated reading time: 2 minutes**")
    
    with st.expander("📖 View English Translation"):
        translations = {
            6: "I have a dog at home. His name is Buddy. Buddy is very loving. I play with him every day. Buddy is a small big dog. He loves to run. We run in the garden. Buddy plays with a ball. He catches the ball and brings it to me. I throw it. Buddy brings it again. We do this over and over.\n\nIn the evening I feed Buddy. He is very happy. He wags his tail. Buddy is my best friend. I love him very much. He shows me how to be patient. I teach Buddy to shake hands. He puts his paw on my hand. He is very clever.\n\nOn weekends we all go to the park. Buddy comes too. There Buddy meets other dogs. He finds new friends. Buddy loves to play with water jets. He gets wet. He jumps on me. I can't stop him. We all laugh.\n\nAt night Buddy sleeps next to my bed. He feels safe. I feel safe. Buddy is part of my family. We love him very much.",
            7: "My favorite sport is football. I play football with my friends every week. I love scoring goals. Our team is very good. We practice together. Our coach is very skilled. He teaches us new skills.\n\nLast week we had a big match. We played against our opponents. The match was very exciting. In the first half we scored one goal. I was very happy about it. But the other team also worked hard. They also scored a goal. The score was one-one.\n\nIn the second half we worked harder. We started running better. We passed the ball well. My friend Sameer gave me an excellent pass. I caught the ball. I ran towards the goal. I shot. The ball went into the net! We won!\n\nAfter the match we were all happy. We celebrate our victory. Our coach was proud of us. He said we did well. Football teaches me teamwork. It helps me make new friends. I am very passionate about football. I want to play every day.",
            8: "I love going to school. My friends are there. I enjoy learning science and math. The teachers are very good. Our classroom is large and bright. There are nice pictures on the walls. We have a library. There are many books in the library.\n\nEvery morning we start with math. I am very fond of mathematics. Our teacher explains things simply. She helps us solve problems. We work together. We learn counting. We learn multiplication. I love solving math problems.\n\nIn science class we do experiments. We conduct tests. Last week we learned about plants. We plant seeds. We water them daily. The seeds grow. It is very interesting. We learned that plants need light and water.\n\nAt lunch time we play outside. I run with my friends. We play hide and seek. Sometimes we read books. The library is quiet and cool. I love reading story books.\n\nAfter school we have homework. I do my homework well. My parents help me. School helps me learn new things. It helps me grow. I love going to school.",
            9: "Last weekend I went to the zoo with my family. There I saw elephants, bears and monkeys. Everything was very interesting. We went early in the morning. The day was new. We went some distance from the city. The zoo is huge. It's like a forest.\n\nFirst we saw elephants. Elephants are very big. They have their long trunks. One elephant was spraying water. It was very fun. Baby elephants were playing near their mother. They are very cute. We spent a lot of time watching them.\n\nNext we saw monkeys. Monkeys jump in trees. They are very fast. Some monkeys are eating bananas. They looked at us. One monkey showed me teeth. It made me smile. Monkeys are very clever.\n\nWe passed the lion cage. The lions are resting. The male lion has a big mane. He looks very strong. Female lionesses are with their cubs. The cubs play with each other. They are soft and cute.\n\nFor lunch we went to the garden cafe. We had sandwiches and fruits. We talked about the whole day. My brother loves monkeys. I love elephants. We all had a fun time.\n\nGoing to the zoo was a great day. We learned about many animals. We saw how they live. I learned about animals from other countries. The zoo helps protect animals. I think we all need to take care of animals.",
            10: "I love reading books. I read adventure and heroic stories. Reading develops my mind and helps me learn new things. Every week I go to the library. I get new books. The library is my favorite place. There are so many books there.\n\nMy favorite book is a heroic story book. It has adventure stories. The main character is a skilled traveler. He travels many places. He meets continents. He finds archaeological ruins. Every chapter is exciting.\n\nI read fiction. I read non-fiction books. I love learning about history. I like learning about science. Books teach me about the world. They teach me about different cultures. When I read, I use my reflection.\n\nSometimes I exchange books with my friends. We read the same book. Then we talk about it. We talk about the story. We talk about the characters. Talking about books is fun. It helps me understand better.\n\nAt night I read before going to sleep. I sit on my bed. I light the lamp. I open a book. I read for thirty minutes. It helps me relax. It calms my mind. Reading is the way to go to sleep.\n\nWhen I grow up, I want to write books. I want to create my own stories. I want to entertain others. I want to teach them new things. Books changed my world. They help me dream. Reading is my beloved hobby.",
            11: "Technology is very important in our lives. I use computers to explore and learn. In the future I want to be a software engineer. I am very passionate about computers. They can do wonderful things. Technology is changing the world.\n\nAt school we have a computer science class. We learn coding. We learn to write programs. My teacher teaches me to write computer code. We make simple games. We make programs to create images. Coding is like solving puzzles.\n\nI practice programming at home. I watch tutorials on the internet. I learn new code languages. I learn Python and JavaScript. I make small projects. I made a calculator. I made a weather app. Each project teaches me new things.\n\nTechnology is changing education. We can learn anything at any time. We have online courses. We have educational videos. We have interactive games. Technology makes learning fun. It makes learning easy.\n\nTechnology is changing communication. We can talk to friends all over the world. We can make video calls. We can send instant messages. Technology connects people. It makes the world smaller.\n\nThe future belongs to technology. Artificial intelligence is opening up. Robots are becoming common. We get self-driving cars. We get smart homes. I want to be part of this future. I want to make good things. I want to use technology to solve problems. I want to make the world a better place.",
            12: "We all must be responsible for protecting the environment. We need to reduce plastic use, plant trees and conserve water. Our planet is our home. We must protect it for future generations. Climate change is a real problem.\n\nThe climate is changing. Temperatures are rising. Ice sheets are melting. Sea level is rising. Extreme weather events are happening. We see more floods. We see more droughts. This is dangerous for our planet.\n\nOur actions can make a significant difference. We can save electricity. We can use renewable energy. We can use less plastic. We can reuse and recycle more. We can plant more trees. Every small action helps.\n\nAt our school we started an environmental project. We plant trees in the school grounds. We created compost bins. We try to reduce plastic. We don't use plastic bottles. We carry reusable bags. Our project is making a difference.\n\nConserving water is also important. Water is a valuable resource. Many people don't have access to clean water. We should not waste water. We can take short showers. We can fix leaks. We can collect rainwater. Conserving water helps the environment.\n\nProtecting wildlife is important. Many animals are facing extinction. Their habitats are being destroyed. We must protect wildlife. We must protect habitats. We can establish national parks and reserves. Every species is important.\n\nProtecting the environment is everyone's responsibility. Our actions determine the future. We must act today. We can start with small things. We can educate our families and friends. Together, we can make a difference. We can build a healthy planet. It is for us and future generations."
        }
        st.write(translations.get(child_age, translations[8]))

with col2:
    st.header("🎙️ Audio Input")
    
    # Tab for upload or record
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🔴 Record Audio"])
    
    audio_data = None
    sample_rate = 16000
    
    with tab1:
        uploaded_file = st.file_uploader("Upload audio file (WAV or MP3)", type=["wav", "mp3"])
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            audio_data = uploaded_file
    
    with tab2:
        st.write("Click the button to start/stop recording")
        
        duration = st.slider("Recording duration (seconds)", 60, 180, 120)
        st.info("⏱️ Recommended: 2 minutes (120 seconds) to read the full paragraph")
        
        if 'recording' not in st.session_state:
            st.session_state.recording = False
        if 'recorded_audio' not in st.session_state:
            st.session_state.recorded_audio = None
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🔴 Start Recording", disabled=st.session_state.recording):
                st.session_state.recording = True
                st.info(f"Recording for {duration} seconds...")
                
                # Record audio
                recording = sd.rec(int(duration * sample_rate), 
                                 samplerate=sample_rate, 
                                 channels=1, 
                                 dtype='float32')
                sd.wait()
                
                st.session_state.recorded_audio = recording
                st.session_state.recording = False
                st.success("Recording completed!")
        
        with col_b:
            if st.button("⏹️ Stop & Save"):
                st.session_state.recording = False
        
        # Display recorded audio
        if st.session_state.recorded_audio is not None:
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            wav.write(temp_file.name, sample_rate, st.session_state.recorded_audio)
            
            st.audio(temp_file.name)
            audio_data = temp_file.name

# Analysis section
if audio_data is not None:
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    with st.spinner("Analyzing speech patterns..."):
        try:
            # Preprocess audio
            audio, sr = preprocess_audio(audio_data)
            
            # Transcribe audio
            if isinstance(audio_data, str):
                transcription = transcribe_audio(audio_data)
            else:
                # Save uploaded file temporarily
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_file.write(audio_data.read())
                temp_file.close()
                transcription = transcribe_audio(temp_file.name)
                os.unlink(temp_file.name)
            
            # Extract features
            features = extract_features(audio, sr, transcription)
            
            # Classify
            result = classify_adhd(features)
            
            # Display results
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("ADHD Probability", f"{result['probability']:.2%}")
            
            with col_res2:
                classification = "ADHD Indicators" if result['adhd'] else "No ADHD Indicators"
                st.metric("Classification", classification)
            
            with col_res3:
                confidence = "High" if abs(result['probability'] - 0.5) > 0.3 else "Medium"
                st.metric("Confidence", confidence)
            
            # Feature details
            with st.expander("📈 Detailed Feature Analysis"):
                st.json(features)
            
            # Transcription
            if transcription:
                with st.expander("📝 Speech Transcription"):
                    st.write(transcription)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            if result['adhd']:
                st.warning("""
                **This screening suggests potential ADHD indicators. Please note:**
                - This is a screening tool, not a diagnostic test
                - Consult a healthcare professional for proper evaluation
                - Early intervention can significantly improve outcomes
                """)
            else:
                st.success("""
                **No significant ADHD indicators detected in this screening.**
                - Continue monitoring speech and behavior patterns
                - Regular developmental check-ups are recommended
                """)
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Please ensure the audio is clear and in a supported format.")

# Footer
st.markdown("---")
st.caption("⚠️ This tool is for screening purposes only and should not replace professional medical diagnosis.")