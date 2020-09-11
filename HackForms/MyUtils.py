from pandas import DataFrame as DF


class MyUtils:

    @staticmethod
    def allInline(top, bottom, df):
        return df[(
            ((df['top'] <= top) & (df['top']+df['height'] >= bottom)) | ((df['top'] >= top) & (df['top'] <= bottom)) | ((
            df['top'] + df['height'] >= top) & (df['top']+df['height'] <= bottom)))]

    @staticmethod
    def topOrBottomInline(top, bottom, df):
        return df[(
        (df['top'] >= top & df['top'] <= bottom) | (df['bottom'] >= top & df['bottom'] <= bottom))]

    @staticmethod
    def strictInline(top, bottom, df):
        return df[(df['top'] <= top & df['bottom'] >= bottom)]

    @staticmethod
    def leftParentOf(currDf, parentDf):
        df = MyUtils.allInline(currDf['top'], currDf[1]+currDf[2], parentDf)
        return df[(df['rightmax'] >= currDf[0]+currDf[3])]

    @staticmethod
    def topParentOf(top, parentDf):
        return parentDf[(parentDf['top'] + parentDf['height'] - top).abs().idxmin()]
