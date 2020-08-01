import re
import numpy as np
from tqdm import tqdm
from load_data import word2idx 

strip_special_chars = re.compile("[^\w0-9 ]+")
replace_list={'òa':'oà','óa':'oá','ỏa':'oả','õa':'oã','ọa':'oạ','òe':'oè','óe':'oé','ỏe':'oẻ','õe':'oẽ','ọe':'oẹ','ùy':'uỳ','úy':'uý','ủy':'uỷ','ũy':'uỹ','ụy':'uỵ','uả':'ủa','ả':'ả','ố':'ố','u´':'ố','ỗ':'ỗ','ồ':'ồ','ổ':'ổ','ấ':'ấ','ẫ':'ẫ','ẩ':'ẩ','ầ':'ầ','ỏ':'ỏ','ề':'ề','ễ':'ễ','ắ':'ắ','ủ':'ủ','ế':'ế','ở':'ở','ỉ':'ỉ','ẻ':'ẻ','àk':' à ','aˋ':'à','iˋ':'ì','ă´':'ắ','ử':'ử','e˜':'ẽ','y˜':'ỹ','a´':'á','👹':'tiêu cực','👻':'tích cực','💃':'tích cực','🤙':'tích cực','👍':'tích cực','💄':'tích cực','💎':'tích cực','💩':'tích cực','😕':'tiêu cực','😱':'tiêu cực','😸':'tích cực','😾':'tiêu cực','🚫':'tiêu cực','\U0001f92c':'tiêu cực','\U0001f9da':'tích cực','\U0001f9e1':'tích cực','🐶':'tích cực','👎':'tiêu cực','😣':'tiêu cực','✨':'tích cực','❣':'tích cực','☀':'tích cực','♥':'tích cực','\U0001f929':'tích cực','like':'tích cực','💌':'tích cực','🤣':'tích cực','🖤':'tích cực','🤤':'tích cực',':(':'tiêu cực','😢':'tiêu cực','❤':'tích cực','😍':'tích cực','😘':'tích cực','😪':'tiêu cực','😊':'tích cực','?':' ? ','😁':'tích cực','💖':'tích cực','😟':'tiêu cực','😭':'tiêu cực','💯':'tích cực','💗':'tích cực','♡':'tích cực','💜':'tích cực','🤗':'tích cực','^^':'tích cực','😨':'tiêu cực','☺':'tích cực','💋':'tích cực','👌':'tích cực','😖':'tiêu cực','😀':'tích cực',':((':'tiêu cực','😡':'tiêu cực','😠':'tiêu cực','😒':'tiêu cực','🙂':'tích cực','😏':'tiêu cực','😝':'tích cực','😄':'tích cực','😙':'tích cực','😤':'tiêu cực','😎':'tích cực','😆':'tích cực','💚':'tích cực','✌':'tích cực','💕':'tích cực','😞':'tiêu cực','😓':'tiêu cực','️🆗️':'tích cực','😉':'tích cực','😂':'tích cực',':v':'tích cực','=))':'tích cực','😋':'tích cực','💓':'tích cực','😐':'tiêu cực',':3':'tích cực','😫':'tiêu cực','😥':'tiêu cực','😃':'tích cực','😬':' 😬 ','😌':' 😌 ','💛':'tích cực','🤝':'tích cực','🎈':'tích cực','😗':'tích cực','🤔':'tiêu cực','😑':'tiêu cực','🔥':'tiêu cực','🙏':'tiêu cực','🆗':'tích cực','😻':'tích cực','💙':'tích cực','💟':'tích cực','😚':'tích cực','❌':'tiêu cực','👏':'tích cực',';)':'tích cực','<3':'tích cực','🌝':'tích cực','🌷':'tích cực','🌸':'tích cực','🌺':'tích cực','🌼':'tích cực','🍓':'tích cực','🐅':'tích cực','🐾':'tích cực','👉':'tích cực','💐':'tích cực','💞':'tích cực','💥':'tích cực','💪':'tích cực','💰':'tích cực','😇':'tích cực','😛':'tích cực','😜':'tích cực','🙃':'tích cực','🤑':'tích cực','\U0001f92a':'tích cực','☹':'tiêu cực','💀':'tiêu cực','😔':'tiêu cực','😧':'tiêu cực','😩':'tiêu cực','😰':'tiêu cực','😳':'tiêu cực','😵':'tiêu cực','😶':'tiêu cực','🙁':'tiêu cực',':))':'tích cực',':)':'tích cực','ô kêi':' ok ','okie':' ok ',' o kê ':' ok ','okey':' ok ','ôkê':' ok ','oki':' ok ',' oke ':' ok ',' okay':' ok ','okê':' ok ',' tks ':' cám ơn ','thks':' cám ơn ','thanks':' cám ơn ','ths':' cám ơn ','thank':' cám ơn ','⭐':'star ','*':'star ','🌟':'star ','🎉':'tích cực','kg ':' không ','not':' không ',' kg ':' không ','"k ':' không ',' kh ':' không ','kô':' không ','hok':' không ',' kp ':' không phải ',' kô ':' không ','"ko ':' không ',' ko ':' không ',' k ':' không ','khong':' không ',' hok ':' không ','he he':'tích cực','hehe':'tích cực','hihi':'tích cực','haha':'tích cực','hjhj':'tích cực',' lol ':'tiêu cực',' cc ':'tiêu cực','cute':' dễ thương ','huhu':'tiêu cực',' vs ':' với ','wa':' quá ','wá':' quá','j':' gì ','“':' ',' sz ':' cỡ ','size':' cỡ ',' đx ':' được ','dk':' được ','dc':' được ','đk':' được ','đc':' được ','authentic':' chuẩn chính hãng ',' aut ':' chuẩn chính hãng ',' auth ':' chuẩn chính hãng ','thick':'tích cực','store':' cửa hàng ','shop':' cửa hàng ','sp':' sản phẩm ','gud':' tốt ','god':' tốt ','wel done':' tốt ','good':' tốt ','gút':' tốt ','sấu':' xấu ','gut':' tốt ',' tot ':' tốt ',' nice ':' tốt ','perfect':'rất tốt','bt':' bình thường ','time':' thời gian ','qá':' quá ',' ship ':' giao hàng ',' m ':' mình ',' mik ':' mình ','ể':'ể','product':'sản phẩm','quality':'chất lượng','chat':' chất ','excelent':'hoàn hảo','bad':'tệ','fresh':' tươi ','sad':' tệ ','date':' hạn sử dụng ','hsd':' hạn sử dụng ','quickly':' nhanh ','quick':' nhanh ','fast':' nhanh ','delivery':' giao hàng ',' síp ':' giao hàng ','beautiful':' đẹp tuyệt vời ',' tl ':' trả lời ',' r ':' rồi ',' shopE ':' cửa hàng ',' order ':' đặt hàng ','chất lg':' chất lượng ',' sd ':' sử dụng ',' dt ':' điện thoại ',' nt ':' nhắn tin ',' sài ':' xài ','bjo':' bao giờ ','thik':' thích ',' sop ':' cửa hàng ',' fb ':' facebook ',' face ':' facebook ',' very ':' rất ','quả ng ':' quảng  ','dep':' đẹp ',' xau ':' xấu ','delicious':' ngon ','hàg':' hàng ','qủa':' quả ','iu':' yêu ','fake':' giả mạo ','trl':'trả lời','><':'tích cực',' por ':' tệ ',' poor ':' tệ ','ib':' nhắn tin ','rep':' trả lời ','fback':' feedback ','fedback':' feedback '}
def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    for k, v in replace_list.items():
        string = string.replace(k, v)
    return re.sub(strip_special_chars, "", string.lower())


def get_sentence_indices(sentence, max_seq_length, _words_list):
    """
    Hàm này dùng để lấy index cho từng từ
    trong câu (không có dấu câu, có thể in hoa)
    Parameters
    ----------
    sentence là câu cần xử lý
    max_seq_length là giới hạn số từ tối đa trong câu
    _words_list là bản sao local của words_list, được truyền vào hàm
    """
    indices = np.zeros((max_seq_length), dtype='int32')

    # Tách câu thành từng tiếng
    words = [word.lower() for word in sentence.split()]

    # Lấy chỉ số của UNK
    unk_idx = word2idx['UNK']

    for idx, word in enumerate(words):
        if idx < max_seq_length:
            if (word in _words_list):
                word_idx = word2idx[word]
            else:
                word_idx = word2idx['UNK']

            indices[idx] = word_idx
        else:
            break

    return indices


def text2ids(df, max_length, _word_list):
    """
    Biến đổi các text trong dataframe thành ma trận index

    Parameters
    ----------
    df: DataFrame
        dataframe chứa các text cần biến đổi
    max_length: int
        độ dài tối đa của một text
    _word_list: numpy.array
        array chứa các từ trong word vectors

    Returns
    -------
    numpy.array
        len(df) x max_length contains indices of text
    """
    ids = np.zeros((len(df), max_length), dtype='int32')
    for idx, text in enumerate(tqdm(df['text'])):
        ids[idx, :] = get_sentence_indices(clean_sentences(text), max_length, _word_list)
    return ids

