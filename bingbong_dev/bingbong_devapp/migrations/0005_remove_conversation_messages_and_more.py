# Generated by Django 5.0.6 on 2024-07-03 02:56

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bingbong_devapp', '0004_member_last_login'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='conversation',
            name='messages',
        ),
        migrations.AlterField(
            model_name='conversation',
            name='member',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='member_convo', to='bingbong_devapp.member'),
        ),
        migrations.CreateModel(
            name='Messages',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('message', models.TextField()),
                ('conversation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='conversations', to='bingbong_devapp.conversation')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='bingbong_devapp.member')),
            ],
        ),
    ]