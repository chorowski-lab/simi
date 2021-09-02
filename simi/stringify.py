from argparse import ArgumentParser
import pathlib


def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('dataset', type=pathlib.Path,
                        help='Path to the dataset for stringification.')
    parser.add_argument('output', type=pathlib.Path,
                        help='Output path')
    return parser.parse_args()


int_to_char = "qwertyuioplkjhgfdsazxcvbnmQWERTYUIOPASDFGHJKLZXCVBNMąężśźćńłóĄĘĆŚŻŹŃÓŁÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſƀƁƂƃƄƅƆƇƈƉƊƋƌƍƎƏƐƑƒƓƔƕƖƗƘƙƚƛƜƝƞƟƠơƢƣƤƥƦƧƨƩƪƫƬƭƮƯưƱƲƳƴƵƶƷƸƹƺƻƼƽƾƿǀǁǂǃǄǅǆǇǈǉǊǋǌǍǎǏǐǑǒǓǔǕǖǗǘǙǚǛǜǝǞǟǠǡǢǣǤǥǦǧǨǩǪǫǬǭǮǯǰǱǲǳǴǵǶǷǸǹǺǻǼǽǾǿȀȁȂȃȄȅȆȇȈȉȊȋȌȍȎȏȐȑȒȓȔȕȖȗȘșȚțȜȝȞȟȠȡȢȣȤȥȦȧȨȩȪȫȬȭȮȯȰȱȲȳΆΈΉΊ΋Ό΍ΎΏΐΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡ΢ΣΤΥΦΧΨΩΪΫάέήίΰαβγδεζ1234567890"
char_to_int = {int_to_char[i] : i for i in range(len(int_to_char))}


def int_array_to_string(array):
    return ''.join(int_to_char[int(x)] for x in array)


def string_to_int_array(s):
    return list(char_to_int[c] for c in s)


def main(args):
    with open(args.output, 'w', encoding='utf8') as out:
        for line in open(args.dataset, 'r', encoding='utf8'):
            fname, data = line.strip().split()
            out.write(f'{fname} {int_array_to_string(data.split(","))}\n')


if __name__ == '__main__':
    args = parseArgs()
    main(args)
