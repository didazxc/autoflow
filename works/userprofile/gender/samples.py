# from .run import trainProcess
# 58 min 10 sec merge_orc_df_rule_before_reduce userprofile/model/gender/latest/sample/merge/sample_20201020.orc
# 61 min 36 sec merge_orc_df userprofile/model/gender/latest/sample/merge/sample_data1000_20201020.orc

# import findspark
#
# findspark.init()


from autoflow.samples.transformers import TagsUserExtractor
from autoflow.utils.table_config import HiveTable
from pyspark.sql import SparkSession

if __name__ == '__main__':

    # spark = SparkSession.builder.master('local[15]') \
    #     .config('spark.driver.memory', '20g') \
    #     .config('spark.driver.maxResultSize', '20g') \
    #     .enableHiveSupport().getOrCreate()
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    # build tags
    male_words = "我去追你,老子想你,我可是+男人,我一大老爷们,你们女孩子,你们女生,晚安老婆,我是处男,我处男,做我的妞,咱媳妇,你们女生,我的老婆,我们男生,老婆大人,我们男的,娶你,我是男人,我是个男人,我一个男的,我一男的,我一个老爷们,作为男人,我老婆,老婆你,媳妇你,俺女朋友,我女朋友,我女友,我媳妇,我带媳妇,俺媳妇,俺老婆,知道了老婆,老婆我想你,老婆我爱你,媳妇我爱你,我的老婆呀,怎么了老婆,我的傻老婆,我的好老婆,哼哼老婆,我没有女友,我没有女朋友"
    female_words = "姐妹们,我的老公,你姐姐我,我可是+女人,我是+妈妈,你们男生,我抛妇产,我剖腹产,我公公,都是姐妹,我和男生,你们男人,俺家老公,我婆婆,我+婆家,我也到预产期,我到预产期,我预产期,我们家男人,我男盆友,本仙女,娶我,我是女的,我们女生,我们女的,谢谢老公,我是女生,我是女士,我一个女的,我一女的,我一个女人,我老公,老公你,我男朋友,我男友,我怀孕,我就怀孕,我闺蜜,我婆婆,老公我爱你,我没老公了,我怀孕,我流产,我大姨妈,我坐月,我妊娠,我子宫,我月经"

    # init SamplesRuleExtractor
    tags = ["male", "female"]
    tag_rules = [f"data_word=:0:{male_words} & (! data_word=:0:{female_words}) & (! data_word=:0:我的男人,我老婆们,我媳妇们,我女朋友们,我女友们) "
                 "& (! apps_list=com.babytree.apps.pregnancy,com.husor.beibei,com.lingan.seeyou,com.baidu.mbaby,com.soft.blued)",
                 f"data_word=:0:{female_words} & (! data_word=:0:{male_words}) & (! apps_list=com.soft.blued)"]

    t = TagsUserExtractor(tag_rules=tag_rules)

    start_date_str = "20201230"
    end_date_str = "20201230"

    sql_data = f"select simei as userid,data from {HiveTable.origin_lines_table} where date between {start_date_str} and {end_date_str}"
    sql_apps = f"select simei as userid,applist as apps from {HiveTable.origin_applist_table} where date between {start_date_str} and {end_date_str}"
    df_data = t.calc_sub_scores(spark.sql(sql_data))
    df_apps = t.calc_sub_scores(spark.sql(sql_apps))
    df = t.calc_tag(df_data.union(df_apps))
    df.write.mode('overwrite').parquet('/tmp/gender_test.parquet')
    # trainProcess.sample(tag_rules=tag_rules)
