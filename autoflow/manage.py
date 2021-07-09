import click
import json
from autoflow.process.user_tags import UserTags


class JsonArrayParamType(click.ParamType):
    name = 'json_array'

    def convert(self, value, param, ctx):
        try:
            arr = json.loads(value)
        except ValueError:
            self.fail('%s is not a valid json string.' % value, param, ctx)
        if len(arr) > 1000:
            self.fail('len(%s) no longer than 1000.' % param, param, ctx)
        else:
            return arr


JSON_ARRAY = JsonArrayParamType()


@click.group()
def cli():
    pass


@cli.command()
@click.argument('filename', type=str)
@click.argument('tags', type=JSON_ARRAY)
@click.argument('tag_rules', type=JSON_ARRAY)
@click.option('--start_date', default=None, type=str)
@click.option('--end_date', default=None, type=str)
@click.option('--mode', default='data', type=str)
def user_tags(filename, tags, tag_rules, start_date, end_date, mode):
    print("start user_tags...")
    print(UserTags.exec_data_market(filename, tags, tag_rules, start_date, end_date, mode))


if __name__ == '__main__':
    cli()
